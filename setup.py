from cx_Freeze import setup, Executable

setup(name="Yolo4_tiny_detection",
	  version="0.1",
	  description="Software detects objects in realtime",
	  executables=[Executable("main.py")]
	  )
#dnn_model failas turi buti ikeltas i exe.win-amd64-3.9 folderi