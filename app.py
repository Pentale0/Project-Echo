import tkinter as tk

from echo.config import Settings
from echo.gui import EchoApp


def main() -> None:
    settings = Settings()
    root = tk.Tk()
    EchoApp(root, settings)
    root.mainloop()


if __name__ == "__main__":
    main()