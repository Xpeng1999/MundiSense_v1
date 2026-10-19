"""Serve the local release page with byte ranges for reliable MP4 chapter seeking."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import re


class ReleaseHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range = None
        path = Path(self.translate_path(self.path))
        value = self.headers.get('Range')
        if not value or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
        if not match or not any(match.groups()) or size == 0:
            return self.invalid_range(size)
        left, right = match.groups()
        start = int(left) if left else max(0, size - int(right))
        end = min(int(right), size - 1) if left and right else size - 1
        if start >= size or start > end:
            return self.invalid_range(size)
        stream = path.open('rb')
        stream.seek(start)
        self.byte_range = (start, end)
        self.send_response(206)
        self.send_header('Content-type', self.guess_type(str(path)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        return stream

    def invalid_range(self, size):
        self.send_response(416)
        self.send_header('Content-Range', f'bytes */{size}')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return None

    def copyfile(self, source, outputfile):
        try:
            if self.byte_range is None:
                return super().copyfile(source, outputfile)
            remaining = self.byte_range[1] - self.byte_range[0] + 1
            while remaining:
                chunk = source.read(min(remaining, 64 * 1024))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers cancel obsolete media requests while seeking.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'docs'
    handler = partial(ReleaseHandler, directory=str(root))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Local: http://127.0.0.1:{args.port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
