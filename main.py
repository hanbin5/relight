"""Start the ID photo app (the Visual Studio startup file).

Same as `uv run --extra fba idphoto`; command-line options such as `--source demo` work here too.
"""

import sys

from idphoto.gui import main

if __name__ == "__main__":
    sys.exit(main())
