#! C:\Users\hanka.jos\Documents\hopefully_working_directory\openMotor\.venv\Scripts\python.exe
import sys
from app import App
from PyQt6.QtCore import Qt

app = App(sys.argv)
sys.exit(app.exec())

# přebrání argumentů objektu App ze souboru app.py
# argumenty jsou -o (output) nazev_souboru_s_vystupem.txt -h (headless, bez GUI) nazev_souboru_s_konfiguracemi.ric
# pr.: -o /home/ufmt/Documents/hopefully_working_directory/output_values.txt -h /home/ufmt/Documents/hopefully_working_directory/motor/simulace_KNSB_1.ric