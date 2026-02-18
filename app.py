from cobra_lite.server import create_app, run_server

app = create_app()

if __name__ == "__main__":
    run_server(app)
