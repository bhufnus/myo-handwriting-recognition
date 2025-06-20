from src.gui import TrainApp

if __name__ == "__main__":
    app = TrainApp(labels=['A', 'B', 'C'], samples_per_class=10, duration_ms=2000)
    app.run() 