#!/usr/bin/env python3
"""
Simple HTTP server for SSH tunnel testing
Listens on port 8888 (uncommon port)
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from datetime import datetime

class TestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SSH Tunnel Test Server</title>
            </head>
            <body>
                <h1>SSH Tunnel Test Successful!</h1>
                <p>Server time: {datetime.now()}</p>
                <p>Your path: {self.path}</p>
                <p>Server port: 8888</p>
                <p>This server is running on the campus network machine.</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode())

        elif self.path == '/api/test':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            response = {
                'status': 'success',
                'message': 'SSH tunnel test API working',
                'timestamp': datetime.now().isoformat(),
                'server_port': 8888
            }
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')

if __name__ == '__main__':
    PORT = 1785
    server = HTTPServer(('0.0.0.0', PORT), TestHandler)
    print(f"Test server running on port {PORT}")
    print(f"Access via SSH tunnel: ssh -L 8080:localhost:{PORT} user@your-server")
    print(f"Then visit: http://localhost:8080")
    server.serve_forever()