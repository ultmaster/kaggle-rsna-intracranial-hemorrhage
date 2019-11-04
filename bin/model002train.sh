model=model002
fold=$1
conf=./conf/${model}.py

python -m src.cnn.main train ${conf} --fold ${fold}
