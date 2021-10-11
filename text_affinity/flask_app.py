import os
import traceback
from flask import Flask

app = Flask(__name__)

if __name__ == '__main__':
    app.run(port='9200', host='0.0.0.0')

'''
def main():
	print (os.listdir("./"))

if __name__ == "__main__":
	main()
'''