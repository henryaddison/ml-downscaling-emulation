import typer

from . import evaluate

app = typer.Typer()
app.add_typer(evaluate.app, name="evaluate")


if __name__ == "__main__":
    app()
