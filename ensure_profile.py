#!/usr/bin/env python3
import subprocess, sys

REQUIRED = '/home/ubuntu/.config/fichte_news'
PORT = '9223'


def main():
    out = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    lines = [l for l in out.splitlines() if 'chromium' in l and f'--remote-debugging-port={PORT}' in l]
    if not lines:
        print(f'PROFILE_CHECK_FAIL: no chromium CDP({PORT}) process')
        sys.exit(2)

    cmd = lines[0]
    if f'--user-data-dir={REQUIRED}' not in cmd:
        print('PROFILE_CHECK_FAIL: wrong profile')
        print('found:', cmd)
        print('required:', REQUIRED)
        sys.exit(3)

    print('PROFILE_CHECK_OK', REQUIRED)


if __name__ == '__main__':
    main()
