PDFLATEX1= pdflatex -synctex=1
#PDFLATEX1= pdflatex 
PDFLATEX2= pdflatex -interaction=batchmode
PDFLATEX3= pdflatex -interaction nonstopmode -halt-on-error -file-line-error 
SH 		 = /bin/bash
ASCRIPT  = /usr/bin/osascript

SOURCE   = plantillaInformeTaller-modMat-UNBOG2017293.tex
BASE     = "$(basename $(SOURCE))"
PDFFILE  = $(BASE).pdf
# Using Biblatex
BIBMNGR = biber
# Using Bibtex
#BIBMNGR = bibtex

default : pdf view 

.PHONY: pdf graphics
pdf: $(SOURCE)
	# run pdflatex and bibtex multiple times to get references right
	$(PDFLATEX1) $(SOURCE) && echo "Done stage 01 \n" && $(PDFLATEX3) $(SOURCE) && echo "Done stage 02 \n" && \
		$(BIBMNGR) $(BASE) && echo "Done stage 03 \n" && $(PDFLATEX2) $(SOURCE) && echo "Done stage 04 \n" && \
		$(BIBMNGR) $(BASE) && echo "Done stage 05 \n" && $(PDFLATEX2) $(SOURCE) && echo "Done stage 06 \n"

.PHONY: view
view: 
	# reload the document in Skim
	#$(SH) skim-view.sh $(BASE)
	open -a Preview.app $(PDFFILE)

.PHONY: git
git:
	git add .
	git commit -m "Project updated by C.Duque as of $$(date +%Y%m%d%H%M%S)"
	git push -u origin main

.PHONY: clean
clean :
	# remove all TeX-generated files in your local directory
	$(RM) -f -- *.aux *.bak *.bbl *.blg *.log *.out *.toc *.tdo _region.* *.bcf *.pdf *.fls *.dvi *.xml

.PHONY: softClean
softClean :
	# remove all TeX-generated files in your local directory
	$(RM) -f -- *.aux *.bak *.bbl *.blg *.log *.out *.toc *.tdo _region.* *.bcf *.fls *.dvi

