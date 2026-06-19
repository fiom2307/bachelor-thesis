.\.venv\Scripts\Activate.ps1
deactivate

pip install -r requirements.txt

python -m scripts.run_csp_lda
python -m scripts.run_eegnet
python -m scripts.run_csp_lda_eegnet

