# Easy Expert Search

## Text Classification and Entity Tagging for Faster Client-Expert Matching
In this repo, you'll find a ready-to-deploy Machine Learning-powered app that uses NLP to solve a client-matching business problem for Gerson-Lehrman Group (GLG).

GLG offers a matchmaking service that pairs clients with industry experts to guide business strategy and decision-making. 

The app is deployed via Docker on AWS Elastic Beanstalk and relies on Streamlit for the front-end to serve Hugging Face transformers and spaCy NER models to automatically categorize incoming client queries and filter GLG's database of experts for the right match. 

## Using this Repo
There are two ways to get started. You may install the dependencies into a virtual environment in your preferred manner, or you may use Docker to handle the environment (recommended). First, clone the repo. Then, in the project directory, run the commands below according to your preferred method. 

Using Docker:
> `$ docker build -t glg-ack .`  
> `$ make docker_okay`  
> `$ make ready`  
> `$ docker run -it -p 8501:80 glg-ack`

Using your own environment:
> Before continuing, you must ensure that `tensorflow==2.6.0` is installed in your environment.  
> Then, open `Makefile` and uncomment the lines under `requirements: test_environment`  
> After, at the command line --  
> `$ make requirements`  
> `$ make ready`  
> `$ cd src/deployment`  
> `$ streamlit run app.py`

Either way, you should have a running Streamlit app that you can navigate to in your browser at localhost:8501.

## Deploying on AWS
Following the above steps should give you the right environment to deploy to AWS Elastic Beanstalk. As long as you have your credentials set up using `aws configure`, you can simply run `eb create glg-ack` at the command line, and Elastic Beanstalk will begin to deploy the app for you. One important change you must make after the environment is available is to change the load balancer protocol from HTTP to TCP on your AWS Elastic Beanstalk console:

1. In the sidebar, click "Configuration" under the environment name ("glg-ack")
2. Scroll down to "Load balancer" and click "Edit"
3. Select the checkbox next to Port 80
4. Click "Actions" -> "Edit"
5. Change the "Listener protocol" to "TCP"
6. Click "Save"
7. Scroll down and click "Apply"

The environment will restart, and once the status returns to "Ok," you should be able to access the app via the provided link.

## Training Your Own Models
If you run the above, you may notice that your classifier doesn't perform that well. This is because the DistilBERT model is not fine-tuned to the task. Please refer to `cgc-1.3` under the notebooks directory for training procedures. If your environment is set up correctly (and ideally you have access to a GPU or are using Google Colab), you should be able to run the notebooks yourself to produce the appropriate DistilBERT. 



File Structure
--------

    ├── LICENSE
    ├── Makefile           
    ├── README.md          
    ├── data               <- Produced via Makefile
    │   ├── processed      <- Transformed data ready for modeling
    │   └── raw            <- Original dataset before transformation
    │
    ├── models             <- Produced via Makefile
    │
    ├── notebooks          <- Jupyter notebooks, with the creator's initials, a number for ordering, and
    │                         a short description
    │
    ├── requirements.txt   <- Project dependencies (does NOT include tensorflow==2.6.0, which is also required)
    │
    ├── setup.py
    └── src
        ├── data           <- Scripts to generate data (assumes you've downloaded Grail QA into data/raw/grail_qa/)
        │   └── make_dataset.py
        │
        ├── deployment     <- Streamlit app
        │   └── app.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        └── models         <- Scripts to get or train models
            ├── get_tokenizer.py
            └── train_model.py
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
