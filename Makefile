

venv/bin/activate: requirements.txt
	virtualenv venv --no-site-packages
	source venv/bin/activate && pip install -r requirements.txt

clean:
	rm -rf venv
