model=model001_large_batch
fold=0
conf=./conf/${model}.py

python -m src.cnn.main train ${conf} --fold ${fold}
