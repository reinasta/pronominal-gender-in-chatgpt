.PHONY: all

base_filename := progen_bias

all: notebook

notebook: ${base_filename}.ipynb
	jupyter nbconvert --execute --to pdf ${base_filename}.ipynb

clean:
	rm -f ${base_filename}.pdf
