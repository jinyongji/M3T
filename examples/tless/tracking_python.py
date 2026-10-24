"""Fail before build/launch when the selected Python lacks the scoring dependency."""
import shlex
import sys


def main():
    try:
        import numpy as np
    except ImportError as error:
        print('T-LESS Python dependency check failed: '+str(error),file=sys.stderr)
        print('Install NumPy into this exact interpreter, then rerun:',file=sys.stderr)
        print(shlex.quote(sys.executable)+" -m pip install --only-binary=:all: 'numpy==1.26.4' -i https://mirrors.bfsu.edu.cn/pypi/web/simple",file=sys.stderr)
        return 2
    print('T-LESS Python: '+sys.executable+'; NumPy: '+np.__version__,flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
