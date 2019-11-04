model=model001
fold=0
conf=./conf/${model}.py

python -m src.cnn.main train ${conf} --fold ${fold}
