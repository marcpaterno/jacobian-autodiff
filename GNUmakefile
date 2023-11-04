all: README.pdf

README.md: README.qmd
	quarto render $<


README.pdf: README.md
	pandoc --from gfm --to pdf -o README.pdf $<


