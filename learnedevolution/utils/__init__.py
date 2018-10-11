import subprocess
import os.path

def git_hash():
    import learnedevolution;
    branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],cwd=os.path.dirname(learnedevolution.__file__)).strip()
    hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],cwd=os.path.dirname(learnedevolution.__file__)).strip()
    return branch.decode("utf-8") +" "+hash.decode("utf-8") ;
