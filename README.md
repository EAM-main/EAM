## EAM: Event-Aware Multimodal Representation Learning for Short Video Fake News Detection

---

## Introduction

The Event-Aware Multimodal (**EAM**) representation learning method that directly learns event semantics from raw multimodal posts, and jointly models both holistic and detailed cues for detection.

![img.png](framework.pdf)


## Dataset
We conduct experiments on two datasets: [FakeSV](https://github.com/ICTMCG/FakeSV) and [FakeTT](https://github.com/ICTMCG/FakingRecipe/blob/main). 

- **Processed Data**:
  - [FakeSV Processed Data](./data/dataset/fake_sv_processed.json)
  - [FakeTT Processed Data](./data/dataset/fake_tt_processed.json)

- **Data Format**:
  ```
    {
       "video_id": "6795508387019869446",   #short video news id
       "annotation": "fake",                #short video news label(real/fake)
       "event": "Chick fil A Offers Free Meals, Shakes and Breakfast Promotions",   #short video news events are summarized by LLM
       "text_description": "We get free meals for a whole year!! #foryou #fyp #chickfila #mypleasure #storytime ", #short video news title
       "video_transcripts": "Waiting outside Chick-fil-A at 4:00 am for a lifetime pass\n Got picked out of 4K people!\nFirst 100",  #short video news transcripts
    }
  ```
##  Environment
conda 4.5.11,Python 3.9.12,pytorch 2.4.1+cu121.For other libs,please refer to the file requirements.txt.
