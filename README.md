# authorship-verification

---

## Updates (January 2026)

**New additions to this repository:**

- **Reorganized Structure**: Files organized into folders (`app/`, `models/`, `training/`, `testing/`, `analysis/`, `logs/`, `visualizations/`, `data/`)
- **New Models**: Added simplified Stylometric + BERT + Ensemble models with improved performance
  - `training/train_stylometric_pan.py` - Train stylometric model
  - `training/finetune_bert_v2.py` - Fine-tune BERT
  - `training/create_ensemble.py` - Create ensemble
  - `testing/test_ensemble_detailed.py` - Evaluate models
- **Streamlit App**: Interactive demo with interpretability in `app/` folder
  - View site with https://same-writer-detector.streamlit.app/
  - Or run locally with: `cd app && streamlit run authorship_app.py`
- **Documentation**: Added `DIRECTORY_STRUCTURE.md`, `CHANGELOG.md`, and `app/README.md`

**Performance Comparison:**

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| **Original Models (from paper):** | | | |
| Fine-tuned BERT | 52.4% | 68.8% | N/A |
| Feature Vector | 62.6% | 65.3% | 0.646 |
| **Original Models (recreated from paper):** | | | |
| Base BERT + Calibration | 63.7% | 58.2% | 0.676 |
| Fine-tuned BERT + Calibration | 70.1% | 71.6% | 0.760 |
| Feature Vector | 58.6% | 57.9% | 0.619 |
| **New Models (Jan 2026):** | | | |
| BERT (fine-tuned, simplified) | 73.9% | 73.8% | 0.821 |
| Stylometric (PAN-style) | 62.2% | 57.1% | 0.665 |
| **Ensemble (BERT + Stylometric)** | **73.9%** | **73.8%** | **0.823** |

**Original code from the paper is preserved** in `featurevector/` and `siamesebert/` folders.

See [CHANGELOG.md](CHANGELOG.md) for detailed changes.

---

Code for authorship verification research project.

Website code can be found at [https://github.com/swan-07/streamlit-av](https://github.com/swan-07/streamlit-av)

Website hosted at [https://same-writer-detector.streamlit.app/](https://same-writer-detector.streamlit.app/)

Paper can be found at [https://swan-07.github.io/assets/Transparent%20Authorship%20Verification.pdf](https://swan-07.github.io/assets/Transparent%20Authorship%20Verification.pdf)

Code detailed in this repo was run in Jupyter Notebooks, and model scripts (everything in the Models section below) was run on  RunPod with an A100 SXM.

Pipeline:

![pipeline](https://github.com/swan-07/authorship-verification/assets/100081902/ad507f59-c3b9-40cb-a0a6-04ef380b4fe8)

Slides: https://docs.google.com/presentation/d/1zG6BA4hjz4E12kYroUOce2GK8M6MuL8UAWxcHkyKqaw/edit?usp=sharing

# Models

[https://github.com/swan-07/authorship-verification/tree/main/featurevector](https://github.com/swan-07/authorship-verification/tree/main/featurevector) has code for implementing the Feature Vector model, modified from the implementation in [https://github.com/janithnw/pan2021_authorship_verification/tree/main](https://github.com/janithnw/pan2021_authorship_verification/tree/main).

Run preprocess.ipynb to preprocess the data (takes a LONG time, I split it into multiple chunks to run at once and combined them in combine.ipynb). 

Run large_train_model.ipynb to create feature vectors and train the model.

Use large_predict.ipynb for predictions and getting important features.

[https://github.com/swan-07/authorship-verification/tree/main/siamesebert/methods](https://github.com/swan-07/authorship-verification/tree/main/siamesebert/methods) has code for implementing the Embedding model.

Run bert.ipynb to train the BERT model (based off the implementation in [https://github.com/JacobTyo/Valla/tree/main](https://github.com/JacobTyo/Valla/tree/main)).

Run logreg.ipynb to fit a logistic regression model to calibrate probability predictions based off of cosine similarity of the embeddings as well as do attention-based highlighting and predictions.
 
# Datasets
Dataset curated can be found on huggingface at [https://huggingface.co/datasets/swan07/authorship-verification](https://huggingface.co/datasets/swan07/authorship-verification)

Code for cleaning and modifying datasets can be found in [https://github.com/swan-07/authorship-verification/blob/main/Authorship_Verification_Datasets.ipynb](https://github.com/swan-07/authorship-verification/blob/main/Authorship_Verification_Datasets.ipynb) and is detailed in paper.

Datasets used to produce the final dataset are:

1. Reuters50

@misc{misc_reuter_50_50_217,
  author       = {Liu,Zhi},
  title        = {{Reuter_50_50}},
  year         = {2011},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5DS42}
}

License: (CC BY 4.0)

2. The Blog Authorship Corpus

@misc{misc_blog_authorship_corpus,
  author       = {J. Schler, M. Koppel, S. Argamon and J. Pennebaker},
  title        = {{Effects of Age and Gender on Blogging}},
  year         = {2006},
  howpublished = {2006 AAAI Spring Symposium on Computational Approaches for Analyzing Weblogs},
  note         = {https://u.cs.biu.ac.il/~schlerj/schler_springsymp06.pdf}
}

License from https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus: The corpus may be freely used for non-commercial research purposes. 

3. Victorian

@misc{misc_victorian_era_authorship_attribution_454,
  author       = {Gungor,Abdulmecit},
  title        = {{Victorian Era Authorship Attribution}},
  year         = {2018},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5SW4H}
}

License: (CC BY 4.0)

4. arXiv

@misc{misc_arXiv_100authors_comp_sci,
  author       = {Moreo, Alejandro},
  title        = {{arXiv abstracts and titles from 1,469 single-authored papers (100 unique authors) in computer science
}},
  year         = {2022},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.7404702}
}

License: (CC BY 4.0)

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

License from https://github.com/bit-ml/Dupin/tree/main: not disclosed

6. British Academic Written English (BAWE)

@misc{20.500.12024/2539,
 title = {British Academic Written English Corpus},
 author = {Nesi, Hilary and Gardner, Sheena and Thompson, Paul and Wickens, Paul},
 url = {http://hdl.handle.net/20.500.12024/2539},
 note = {Oxford Text Archive},
 copyright = {Distributed by the University of Oxford under a Creative Commons Attribution-{NonCommercial}-{ShareAlike} 3.0 Unported License.},
 year = {2008} }
 
 License from https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2539: (CC BY-NC-SA 3.0) 

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

License from https://umlt.infotech.monash.edu/?page_id=266: not disclosed

8. PAN11

@misc{misc_pan11-author-identification-corpora,
  author       = {Argamon, Shlomo and Juola, Patrick},
  title        = {{PAN11 Author Identification: Attribution}},
  year         = {2011},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.3713245}
}

License: not disclosed

9. PAN13

@misc{misc_pan13-authorship-verification-test-and-training,
  author       = {Juola, Patrick and Stamatatos, Efstathios},
  title        = {{PAN13 Author Identification: Verification}},
  year         = {2013},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.3715998}
}

License: not disclosed

10. PAN14

@misc{misc_pan14-authorship-verification-test-and-training,
  author = {Stamatatos,  Efstathios and Daelemans,  Walter and Verhoeven,  Ben and Potthast,  Martin and Stein,  Benno and Juola,  Patrick and A. Sanchez-Perez,  Miguel and Barrón-Cedeño,  Alberto},
  title        = {{PAN14 Author Identification: Verification}},
  year         = {2014},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.3716032}
}

License: not disclosed

11. PAN15

@misc{misc_pan15-authorship-verification-test-and-training,
  author = {Stamatatos,  Efstathios and Daelemans Daelemans amd Ben Verhoeven,  Walter and Juola,  Patrick and López-López,  Aurelio and Potthast,  Martin and Stein,  Benno},
  title        = {{PAN15 Author Identification: Verification}},
  year         = {2015},
  howpublished = {Zenodo},
  note         = {{DOI}: https://doi.org/10.5281/zenodo.3737563}
}

License: not disclosed

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

License from https://github.com/bit-ml/Dupin/tree/main: not disclosed


Datasets were cleaned, named entities were replaced with their general type in all except PAN14, PAN15, and PAN20, and datasets were restructured into dataframes with columns |text1|text2|same| where a value of 0 in same meant the two texts had different authors, while a value of 1 meant the two texts had the same author.

All datasets were split into train/test/verification, keeping the splits if given (see paper for specifics) and otherwise using a 0.7:0.15:0.15 split.
