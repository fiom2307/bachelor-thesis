.\.venv\Scripts\Activate.ps1
deactivate

pip install -r requirements.txt

python -m scripts.compare_csp_lda_eegnet

