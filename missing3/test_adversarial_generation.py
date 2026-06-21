import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import csv
from tqdm import tqdm

from models1 import Mdfuse
from fmcnet_modules import FMCEnhancedDrFuse, FMCCompensationLoss
from enhanced_adversarial_generation import EnhancedAdversarialFramework
from src.huston2013_dataloader import huston2013_multi_dataloader
from config_enhanced import args


class AdversarialGenerationTester:
    """
    Comprehensive testing framework for adversarial feature generation
    """

    def __init__(self, args, model, adversarial_framework, device='cuda'):
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.adversarial_framework = adversarial_framework.to(device) if adversarial_framework else None

        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.feature_similarity = nn.CosineSimilarity(dim=1)

        # Results storage
        self.test_results = {}

    def extract_features(self, model, x1, x2):
        """Extract intermediate features from model"""
        if isinstance(model, FMCEnhancedDrFuse):
            # For FMC-enhanced model
            with torch.no_grad():
                output, features = model(x1, x2)
                return features
        else:
            # For standard model - extract features manually
            if hasattr(model, 'sfd_module'):
                f_sh_1, f_sh_2, f_sp_1, f_sp_2 = model.sfd_module(x1, x2)
                return {
                    'hsi_shared': f_sh_1,
                    'lidar_shared': f_sh_2,
                    'hsi_specific': f_sp_1,
                    'lidar_specific': f_sp_2
                }
            else:
                # Fallback - return dummy features
                batch_size = x1.size(0)
                feature_size = 7  # 7x7 spatial size
                return {
                    'hsi_shared': torch.zeros(batch_size, 64, feature_size, feature_size).to(self.device),
                    'lidar_shared': torch.zeros(batch_size, 64, feature_size, feature_size).to(self.device),
                    'hsi_specific': torch.zeros(batch_size, 64, feature_size, feature_size).to(self.device),
                    'lidar_specific': torch.zeros(batch_size, 64, feature_size, feature_size).to(self.device)
                }

    def test_generation_quality(self, test_loader, missing_scenarios):
        """
        Test quality of generated features under different missing modality scenarios

        Args:
            test_loader: Test data loader
            missing_scenarios: List of missing modality scenarios to test

        Returns:
            Dictionary containing generation quality metrics
        """
        print("Testing generation quality...")

        generation_results = {}

        for scenario in missing_scenarios:
            print(f"Testing scenario: {scenario['name']}")

            scenario_results = {
                'reconstruction_errors': [],
                'feature_similarities': [],
                'attention_weights': [],
                'generation_times': [],
                'classification_accuracy': 0
            }

            self.model.eval()
            if self.adversarial_framework:
                self.adversarial_framework.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc=f"Testing {scenario['name']}")):
                    if isinstance(data, (list, tuple)) and len(data) == 2:
                        x1, x2 = data[0].to(self.device), data[1].to(self.device)
                    else:
                        x1 = data.to(self.device)
                        x2 = torch.zeros_like(x1)

                    targets = targets.to(self.device)
                    batch_size = x1.size(0)

                    # Create missing modality mask
                    if scenario['name'] == 'random_missing':
                        missing_mask = torch.rand(batch_size, 2) > scenario['missing_rate']
                        # Ensure at least one modality
                        missing_all = (missing_mask.sum(dim=1) == 0)
                        if missing_all.any():
                            rand_modality = torch.randint(0, 2, (missing_all.sum(),))
                            missing_mask[missing_all, rand_modality] = True
                    else:
                        missing_mask = torch.tensor([scenario['missing_mask']] * batch_size)

                    missing_mask = missing_mask.float().to(self.device)

                    # Extract real features for comparison
                    real_features = self.extract_features(self.model, x1, x2)

                    # Generate missing features
                    if self.adversarial_framework and missing_mask.sum() < batch_size * 2:
                        # Create available features dictionary
                        available_features = {}
                        for i, (key, value) in enumerate(real_features.items()):
                            if missing_mask[0, i % 2] == 1:  # Modality available
                                available_features[key] = value

                        # Time generation process
                        start_time = time.time()
                        generated_features, generation_info = self.adversarial_framework.generate_missing_features(
                            available_features, missing_mask[0:1]  # Use first sample mask
                        )
                        generation_time = time.time() - start_time

                        scenario_results['generation_times'].append(generation_time)

                        # Store attention weights if available
                        if generation_info:
                            for modality in ['hsi', 'lidar']:
                                if modality in generation_info:
                                    info = generation_info[modality]
                                    if 'spatial_attention' in info:
                                        scenario_results['attention_weights'].append({
                                            'modality': modality,
                                            'spatial_attention': info['spatial_attention'].cpu().numpy(),
                                            'channel_attention': info.get('channel_attention', None).cpu().numpy() if info.get('channel_attention') is not None else None,
                                            'fmc_attention': info.get('fmc_attention', None)
                                        })

                        # Evaluate generation quality
                        for modality in ['hsi', 'lidar']:
                            if f'{modality}_specific' in generated_features and f'{modality}_specific' in real_features:
                                real_feat = real_features[f'{modality}_specific']
                                gen_feat = generated_features[f'{modality}_specific']

                                # Reconstruction error
                                recon_error = self.reconstruction_loss(gen_feat, real_feat).item()
                                scenario_results['reconstruction_errors'].append({
                                    'modality': modality,
                                    'error': recon_error,
                                    'batch_idx': batch_idx
                                })

                                # Feature similarity
                                similarity = self.feature_similarity(
                                    gen_feat.view(gen_feat.size(0), -1),
                                    real_feat.view(real_feat.size(0), -1)
                                ).mean().item()
                                scenario_results['feature_similarities'].append({
                                    'modality': modality,
                                    'similarity': similarity,
                                    'batch_idx': batch_idx
                                })

                    # Test classification with generated features
                    if isinstance(self.model, FMCEnhancedDrFuse):
                        # Use FMC-enhanced model
                        outputs, _ = self.model(x1, x2, modality_mask=missing_mask)
                    else:
                        # Use standard model
                        outputs = self.model(x1, x2)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            # Calculate classification accuracy
            scenario_results['classification_accuracy'] = 100. * correct / total
            generation_results[scenario['name']] = scenario_results

            print(f"{scenario['name']} - Classification Accuracy: {scenario_results['classification_accuracy']:.2f}%")
            if scenario_results['reconstruction_errors']:
                avg_recon_error = np.mean([r['error'] for r in scenario_results['reconstruction_errors']])
                print(f"Average Reconstruction Error: {avg_recon_error:.4f}")
            if scenario_results['feature_similarities']:
                avg_similarity = np.mean([s['similarity'] for s in scenario_results['feature_similarities']])
                print(f"Average Feature Similarity: {avg_similarity:.4f}")

        return generation_results

    def test_adversarial_robustness(self, test_loader, perturbation_levels):
        """
        Test robustness of adversarial generation under input perturbations

        Args:
            test_loader: Test data loader
            perturbation_levels: List of noise levels to test

        Returns:
            Dictionary containing robustness metrics
        """
        print("Testing adversarial robustness...")

        robustness_results = {}

        for noise_level in perturbation_levels:
            print(f"Testing noise level: {noise_level}")

            noise_results = {
                'generation_stability': [],
                'feature_consistency': [],
                'classification_accuracy': 0
            }

            self.model.eval()
            if self.adversarial_framework:
                self.adversarial_framework.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc=f"Noise {noise_level}")):
                    if isinstance(data, (list, tuple)) and len(data) == 2:
                        x1, x2 = data[0].to(self.device), data[1].to(self.device)
                    else:
                        x1 = data.to(self.device)
                        x2 = torch.zeros_like(x1)

                    targets = targets.to(self.device)

                    # Add noise to inputs
                    noisy_x1 = x1 + torch.randn_like(x1) * noise_level
                    noisy_x2 = x2 + torch.randn_like(x2) * noise_level

                    # Create missing modality mask (simulate HSI missing)
                    missing_mask = torch.tensor([[0, 1]] * x1.size(0)).float().to(self.device)

                    # Extract clean and noisy features
                    clean_features = self.extract_features(self.model, x1, x2)
                    noisy_features = self.extract_features(self.model, noisy_x1, noisy_x2)

                    # Generate features for both clean and noisy inputs
                    if self.adversarial_framework:
                        # Create available features dictionary
                        available_features = {'lidar_shared': noisy_features['lidar_shared'],
                                            'lidar_specific': noisy_features['lidar_specific']}

                        # Generate HSI features
                        generated_clean, _ = self.adversarial_framework.generators['hsi'](
                            clean_features['lidar_shared'],
                            clean_features.get('lidar_specific')
                        )
                        generated_noisy, _ = self.adversarial_framework.generators['hsi'](
                            noisy_features['lidar_shared'],
                            noisy_features.get('lidar_specific')
                        )

                        # Measure generation stability
                        stability = self.reconstruction_loss(generated_clean, generated_noisy).item()
                        noise_results['generation_stability'].append(stability)

                        # Measure feature consistency
                        if 'hsi_specific' in clean_features:
                            consistency = self.feature_similarity(
                                generated_noisy.view(generated_noisy.size(0), -1),
                                clean_features['hsi_specific'].view(clean_features['hsi_specific'].size(0), -1)
                            ).mean().item()
                            noise_results['feature_consistency'].append(consistency)

                    # Test classification with noisy inputs
                    if isinstance(self.model, FMCEnhancedDrFuse):
                        outputs, _ = self.model(noisy_x1, noisy_x2, modality_mask=missing_mask)
                    else:
                        outputs = self.model(noisy_x1, noisy_x2)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            # Calculate accuracy
            noise_results['classification_accuracy'] = 100. * correct / total
            robustness_results[f'noise_{noise_level}'] = noise_results

            print(f"Noise {noise_level} - Classification Accuracy: {noise_results['classification_accuracy']:.2f}%")
            if noise_results['generation_stability']:
                avg_stability = np.mean(noise_results['generation_stability'])
                print(f"Average Generation Stability: {avg_stability:.4f}")
            if noise_results['feature_consistency']:
                avg_consistency = np.mean(noise_results['feature_consistency'])
                print(f"Average Feature Consistency: {avg_consistency:.4f}")

        return robustness_results

    def visualize_generation_results(self, generation_results, save_path='../output/figures/'):
        """
        Visualize generation quality results

        Args:
            generation_results: Results from generation quality testing
            save_path: Path to save visualizations
        """
        print("Visualizing generation results...")

        os.makedirs(save_path, exist_ok=True)

        # Plot reconstruction errors
        plt.figure(figsize=(15, 10))

        # Reconstruction errors by scenario
        plt.subplot(2, 3, 1)
        scenarios = []
        errors = []
        for scenario_name, results in generation_results.items():
            if results['reconstruction_errors']:
                avg_error = np.mean([r['error'] for r in results['reconstruction_errors']])
                scenarios.append(scenario_name)
                errors.append(avg_error)

        if scenarios and errors:
            plt.bar(scenarios, errors)
            plt.title('Average Reconstruction Error by Scenario')
            plt.ylabel('MSE Loss')
            plt.xticks(rotation=45)

        # Feature similarities by scenario
        plt.subplot(2, 3, 2)
        similarities = []
        for scenario_name, results in generation_results.items():
            if results['feature_similarities']:
                avg_similarity = np.mean([s['similarity'] for s in results['feature_similarities']])
                similarities.append(avg_similarity)

        if scenarios and similarities:
            plt.bar(scenarios, similarities)
            plt.title('Average Feature Similarity by Scenario')
            plt.ylabel('Cosine Similarity')
            plt.xticks(rotation=45)

        # Classification accuracy by scenario
        plt.subplot(2, 3, 3)
        accuracies = [results['classification_accuracy'] for results in generation_results.values()]
        if scenarios and accuracies:
            plt.bar(scenarios, accuracies)
            plt.title('Classification Accuracy by Scenario')
            plt.ylabel('Accuracy (%)')
            plt.xticks(rotation=45)

        # Generation time distribution
        plt.subplot(2, 3, 4)
        all_times = []
        for results in generation_results.values():
            all_times.extend(results['generation_times'])

        if all_times:
            plt.hist(all_times, bins=20, alpha=0.7)
            plt.title('Generation Time Distribution')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')

        # Attention weight visualization (if available)
        plt.subplot(2, 3, 5)
        attention_data = []
        for scenario_name, results in generation_results.items():
            for attention_info in results['attention_weights']:
                if attention_info['spatial_attention'] is not None:
                    spatial_attn = attention_info['spatial_attention'].mean()
                    attention_data.append({
                        'scenario': scenario_name,
                        'modality': attention_info['modality'],
                        'spatial_attention': spatial_attn
                    })

        if attention_data:
            import pandas as pd
            df = pd.DataFrame(attention_data)
            sns.boxplot(data=df, x='scenario', y='spatial_attention', hue='modality')
            plt.title('Spatial Attention Weights')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'generation_quality_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generation quality visualizations saved to {save_path}")

    def save_test_results(self, results, save_path='../output/logs/'):
        """
        Save test results to CSV files

        Args:
            results: Dictionary containing test results
            save_path: Path to save results
        """
        print("Saving test results...")

        os.makedirs(save_path, exist_ok=True)

        # Save generation quality results
        if 'generation_quality' in results:
            gen_results = results['generation_quality']

            # Save reconstruction errors
            recon_errors_file = os.path.join(save_path, 'reconstruction_errors.csv')
            with open(recon_errors_file, 'w', newline='') as csvfile:
                fieldnames = ['scenario', 'modality', 'error', 'batch_idx']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for scenario_name, scenario_results in gen_results.items():
                    for error_info in scenario_results['reconstruction_errors']:
                        writer.writerow({
                            'scenario': scenario_name,
                            'modality': error_info['modality'],
                            'error': error_info['error'],
                            'batch_idx': error_info['batch_idx']
                        })

            # Save feature similarities
            similarities_file = os.path.join(save_path, 'feature_similarities.csv')
            with open(similarities_file, 'w', newline='') as csvfile:
                fieldnames = ['scenario', 'modality', 'similarity', 'batch_idx']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for scenario_name, scenario_results in gen_results.items():
                    for sim_info in scenario_results['feature_similarities']:
                        writer.writerow({
                            'scenario': scenario_name,
                            'modality': sim_info['modality'],
                            'similarity': sim_info['similarity'],
                            'batch_idx': sim_info['batch_idx']
                        })

            # Save summary statistics
            summary_file = os.path.join(save_path, 'test_summary.csv')
            with open(summary_file, 'w', newline='') as csvfile:
                fieldnames = ['scenario', 'classification_accuracy', 'avg_reconstruction_error',
                            'avg_feature_similarity', 'avg_generation_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for scenario_name, scenario_results in gen_results.items():
                    avg_recon_error = np.mean([r['error'] for r in scenario_results['reconstruction_errors']]) if scenario_results['reconstruction_errors'] else 0
                    avg_similarity = np.mean([s['similarity'] for s in scenario_results['feature_similarities']]) if scenario_results['feature_similarities'] else 0
                    avg_gen_time = np.mean(scenario_results['generation_times']) if scenario_results['generation_times'] else 0

                    writer.writerow({
                        'scenario': scenario_name,
                        'classification_accuracy': scenario_results['classification_accuracy'],
                        'avg_reconstruction_error': avg_recon_error,
                        'avg_feature_similarity': avg_similarity,
                        'avg_generation_time': avg_gen_time
                    })

        print(f"Test results saved to {save_path}")

    def run_comprehensive_test(self, test_loader):
        """
        Run comprehensive testing of adversarial generation

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary containing all test results
        """
        print("Starting comprehensive adversarial generation testing...")

        # Define test scenarios
        missing_scenarios = [
            {'name': 'complete', 'missing_mask': [1, 1]},
            {'name': 'hsi_missing', 'missing_mask': [0, 1]},
            {'name': 'lidar_missing', 'missing_mask': [1, 0]},
            {'name': 'random_missing', 'missing_rate': 0.5}
        ]

        perturbation_levels = [0.01, 0.05, 0.1, 0.2]

        # Run tests
        print("1. Testing generation quality...")
        generation_quality = self.test_generation_quality(test_loader, missing_scenarios)

        print("\n2. Testing adversarial robustness...")
        robustness_results = self.test_adversarial_robustness(test_loader, perturbation_levels)

        # Compile results
        all_results = {
            'generation_quality': generation_quality,
            'robustness': robustness_results,
            'test_config': {
                'missing_scenarios': missing_scenarios,
                'perturbation_levels': perturbation_levels,
                'model_type': type(self.model).__name__,
                'adversarial_framework': self.adversarial_framework is not None
            }
        }

        # Visualize results
        self.visualize_generation_results(generation_quality)

        # Save results
        self.save_test_results(all_results)

        print("Comprehensive testing completed!")
        return all_results


def main():
    """
    Main function to run adversarial generation tests
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    print("Loading test data...")
    test_loader = huston2013_multi_dataloader(train=False, args=args)

    # Setup models
    print("Setting up models...")
    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    # Use FMC-enhanced model for testing
    model = FMCEnhancedDrFuse(
        args, modality_1_channel, modality_2_channel,
        feature_dim=args.feature_dim if hasattr(args, 'feature_dim') else 128,
        shared_dim=args.shared_dim if hasattr(args, 'shared_dim') else 64,
        specific_dim=args.specific_dim if hasattr(args, 'specific_dim') else 64
    )

    # Create adversarial framework
    adversarial_framework = EnhancedAdversarialFramework(
        shared_dim=args.shared_dim if hasattr(args, 'shared_dim') else 64,
        specific_dim=args.specific_dim if hasattr(args, 'specific_dim') else 64,
        feature_dim=args.feature_dim if hasattr(args, 'feature_dim') else 128,
        num_modalities=2
    )

    # Create tester
    tester = AdversarialGenerationTester(args, model, adversarial_framework, device)

    # Run comprehensive test
    results = tester.run_comprehensive_test(test_loader)

    # Print summary
    print("\n" + "="*50)
    print("ADVERSARIAL GENERATION TEST SUMMARY")
    print("="*50)

    if 'generation_quality' in results:
        print("\nGeneration Quality Results:")
        for scenario, metrics in results['generation_quality'].items():
            print(f"  {scenario}:")
            print(f"    Classification Accuracy: {metrics['classification_accuracy']:.2f}%")
            if metrics['reconstruction_errors']:
                avg_error = np.mean([r['error'] for r in metrics['reconstruction_errors']])
                print(f"    Average Reconstruction Error: {avg_error:.4f}")
            if metrics['feature_similarities']:
                avg_sim = np.mean([s['similarity'] for s in metrics['feature_similarities']])
                print(f"    Average Feature Similarity: {avg_sim:.4f}")

    if 'robustness' in results:
        print("\nRobustness Results:")
        for noise_level, metrics in results['robustness'].items():
            print(f"  {noise_level}:")
            print(f"    Classification Accuracy: {metrics['classification_accuracy']:.2f}%")
            if metrics['generation_stability']:
                avg_stability = np.mean(metrics['generation_stability'])
                print(f"    Average Generation Stability: {avg_stability:.4f}")
            if metrics['feature_consistency']:
                avg_consistency = np.mean(metrics['feature_consistency'])
                print(f"    Average Feature Consistency: {avg_consistency:.4f}")

    print("\nTesting completed successfully!")


if __name__ == '__main__':
    import time
    main()