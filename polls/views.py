import pandas as pd
from django.shortcuts import get_object_or_404, render
from .models import Question

import pickle
import numpy as np


def index(request):
    """
    Displays the latest 5 polls questions in the system
    """

    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    context = {
        'latest_question_list': latest_question_list
    }

    return render(request, 'polls/index.html', context)


def home(request):
    with open('E:/tcc_website/polls/toxic-model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    toxic = ''
    tmp = ''
    if 'comment' in request.POST:
        toxic = 1
        tmp = request.POST.get('comment')

        prediction = predict_single([tmp], dv, model)
        toxic = int(prediction[0])

    context = {'toxic': toxic, 'tmp': tmp}

    return render(request, 'polls/home.html', context)


def predict_single(text, dv, model):
    X = dv.transform(text)
    y_pred = model.predict(X)
    return y_pred


def predict(file, dv, model):
    df = pd.read_csv(file)
    X = dv.transform(df)
    y = model(X)[:, 1]
    df['y'] = y
    df.to_csv(index=False)
    return 0
