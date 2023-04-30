from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from .apps import ApiConfig


class PredictAbsenteeism(APIView):
    def post(self, request):
        request = request.data
        reason = request["reason_for_absence"]
        transportation = request["transportation_expense"]
        disciplinary_failure = request["disciplinary_failure"]
        has_son = request["son"]
        class_model = ApiConfig.model
        prediction = class_model.predict([[reason, transportation, disciplinary_failure, has_son]])

        # A prediction of 1 means that the employee will likely be an absentee while a prediction of 0 mean he won't
        prediction = "Prone to Absenteeism " if prediction[0] == 1 else "Not prone to Absenteeism"
        response_dict = {"predicted_character": prediction}
        # response_dict = {"g": "great"}
        print(response_dict)
        return Response(response_dict, status=200)


absenteeism_predict = PredictAbsenteeism.as_view()


