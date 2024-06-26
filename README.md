# authorship-verification
Code for authorship verification research project. (Code is a little messy right now, see paper for overview)

Website code can be found at https://github.com/swan-07/streamlit-av
Website hosted at https://same-writer-detector.streamlit.app/

# Datasets
Dataset curated can be found on huggingface at [https://huggingface.co/datasets/swan07/authorship-verification](url)

Code for cleaning and modifying datasets can be found in [https://github.com/swan-07/authorship-verification/blob/main/Authorship_Verification_Datasets.ipynb](url) and is detailed in paper.

Datasets used to produce the final dataset are:

1. Reuters50

@misc{misc_reuter_50_50_217,
  author       = {Liu,Zhi},
  title        = {{Reuter_50_50}},
  year         = {2011},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5DS42}
}

2. The Blog Authorship Corpus

@misc{misc_blog_authorship_corpus,
  author       = {J. Schler, M. Koppel, S. Argamon and J. Pennebaker},
  title        = {{Effects of Age and Gender on Blogging}},
  year         = {2006},
  howpublished = {2006 AAAI Spring Symposium on Computational Approaches for Analyzing Weblogs},
  note         = {https://u.cs.biu.ac.il/~schlerj/schler_springsymp06.pdf}
}

3. Victorian

@misc{misc_victorian_era_authorship_attribution_454,
  author       = {Gungor,Abdulmecit},
  title        = {{Victorian Era Authorship Attribution}},
  year         = {2018},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5SW4H}
}

4. arXiv

@misc{misc_arXiv_100authors_comp_sci,
  author       = {Moreo, Alejandro},
  title        = {{arXiv abstracts and titles from 1,469 single-authored papers (100 unique authors) in computer science
}},
  year         = {2022},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.7404702}
}

5. DarkReddit

@article{DBLP:journals/corr/abs-2112-05125,
  author    = {Andrei Manolache and
               Florin Brad and
               Elena Burceanu and
               Antonio Barbalau and
               Radu Tudor Ionescu and
               Marius Popescu},
  title     = {Transferring BERT-like Transformers' Knowledge for Authorship Verification},
  journal   = {CoRR},
  volume    = {abs/2112.05125},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.05125},
  eprinttype = {arXiv},
  eprint    = {2112.05125},
  timestamp = {Mon, 13 Dec 2021 17:51:48 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-05125.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{Kestemont2020OverviewOT,
  author    = {Mike Kestemont and
               Enrique Manjavacas and
               Ilia Markov and
               Janek Bevendorff and
               Matti Wiegmann and
               Efstathios Stamatatos and
               Martin Potthast and
               Benno Stein},
  editor    = {Linda Cappellato and
               Carsten Eickhoff and
               Nicola Ferro and
               Aur{\'{e}}lie N{\'{e}}v{\'{e}}ol},
  title     = {Overview of the Cross-Domain Authorship Verification Task at {PAN}
               2020},
  booktitle = {Working Notes of {CLEF} 2020 - Conference and Labs of the Evaluation
               Forum, Thessaloniki, Greece, September 22-25, 2020},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2696},
  publisher = {CEUR-WS.org},
  year      = {2020},
  url       = {http://ceur-ws.org/Vol-2696/paper\_264.pdf},
  timestamp = {Tue, 27 Oct 2020 17:12:48 +0100},
  biburl    = {https://dblp.org/rec/conf/clef/KestemontMMBWSP20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

6. British Academic Written English (BAWE)

@misc{20.500.12024/2539,
 title = {British Academic Written English Corpus},
 author = {Nesi, Hilary and Gardner, Sheena and Thompson, Paul and Wickens, Paul},
 url = {http://hdl.handle.net/20.500.12024/2539},
 note = {Oxford Text Archive},
 copyright = {Distributed by the University of Oxford under a Creative Commons Attribution-{NonCommercial}-{ShareAlike} 3.0 Unported License.},
 year = {2008} }

7. IMDB62

@article{seroussi2014authorship,
  title={Authorship attribution with topic models},
  author={Seroussi, Yanir and Zukerman, Ingrid and Bohnert, Fabian},
  journal={Computational Linguistics},
  volume={40},
  number={2},
  pages={269--310},
  year={2014},
  publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}

8. PAN11

@misc{misc_pan11-author-identification-corpora,
  author       = {Argamon, Shlomo and Juola, Patrick},
  title        = {{PAN11 Author Identification: Attribution}},
  year         = {2011},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.7404702}
}

9. PAN13

@misc{misc_pan13-authorship-verification-test-and-training,
  author       = {Juola, Patrick and Stamatatos, Efstathios},
  title        = {{PAN13 Author Identification: Verification}},
  year         = {2013},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.3715998}
}

10. PAN14

@misc{misc_pan14-authorship-verification-test-and-training,
  author = {Stamatatos,  Efstathios and Daelemans,  Walter and Verhoeven,  Ben and Potthast,  Martin and Stein,  Benno and Juola,  Patrick and A. Sanchez-Perez,  Miguel and Barrón-Cedeño,  Alberto},
  title        = {{PAN14 Author Identification: Verification}},
  year         = {2014},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.3716032}
}

11. PAN15

@misc{misc_pan15-authorship-verification-test-and-training,
  author = {Stamatatos,  Efstathios and Daelemans Daelemans amd Ben Verhoeven,  Walter and Juola,  Patrick and López-López,  Aurelio and Potthast,  Martin and Stein,  Benno},
  title        = {{PAN15 Author Identification: Verification}},
  year         = {2015},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.3737563}
}

12. PAN20

@Article{stein:2020k,
  author =              {Sebastian Bischoff and Niklas Deckers and Marcel Schliebs and Ben Thies and Matthias Hagen and Efstathios Stamatatos and Benno Stein and Martin Potthast},
  journal =             {CoRR},
  month =               may,
  title =               {{The Importance of Suppressing Domain Style in Authorship Analysis}},
  url =                 {https://arxiv.org/abs/2005.14714},
  volume =              {abs/2005.14714},
  year =                2020
}

using the open-set unseen all split from 
@article{DBLP:journals/corr/abs-2112-05125,
  author    = {Andrei Manolache and
               Florin Brad and
               Elena Burceanu and
               Antonio Barbalau and
               Radu Tudor Ionescu and
               Marius Popescu},
  title     = {Transferring BERT-like Transformers' Knowledge for Authorship Verification},
  journal   = {CoRR},
  volume    = {abs/2112.05125},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.05125},
  eprinttype = {arXiv},
  eprint    = {2112.05125},
  timestamp = {Mon, 13 Dec 2021 17:51:48 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-05125.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{Kestemont2020OverviewOT,
  author    = {Mike Kestemont and
               Enrique Manjavacas and
               Ilia Markov and
               Janek Bevendorff and
               Matti Wiegmann and
               Efstathios Stamatatos and
               Martin Potthast and
               Benno Stein},
  editor    = {Linda Cappellato and
               Carsten Eickhoff and
               Nicola Ferro and
               Aur{\'{e}}lie N{\'{e}}v{\'{e}}ol},
  title     = {Overview of the Cross-Domain Authorship Verification Task at {PAN}
               2020},
  booktitle = {Working Notes of {CLEF} 2020 - Conference and Labs of the Evaluation
               Forum, Thessaloniki, Greece, September 22-25, 2020},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2696},
  publisher = {CEUR-WS.org},
  year      = {2020},
  url       = {http://ceur-ws.org/Vol-2696/paper\_264.pdf},
  timestamp = {Tue, 27 Oct 2020 17:12:48 +0100},
  biburl    = {https://dblp.org/rec/conf/clef/KestemontMMBWSP20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

Datasets were cleaned, named entities were replaced with their general type in all except PAN14, PAN15, and PAN20, and datasets were restructured into dataframes with columns |text1|text2|same| where a value of 0 in same meant the two texts had different authors, while a value of 1 meant the two texts had the same author.

All datasets were split into train/test/verification, keeping the splits if given (see paper for specifics) and otherwise using a 0.7:0.15:0.15 split.
