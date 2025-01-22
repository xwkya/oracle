from pathlib import Path


class CoreUtils:
    @staticmethod
    def get_root() -> Path:
        """
        Returns a reference to the root path of the project.
        TODO: Cache this?
        """
        path = Path(__file__).resolve().parent
        while '.env' not in [f.name for f in path.iterdir()]:
            path = path.parent

        return path

