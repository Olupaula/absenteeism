from django.urls import path
from . import views
urlpatterns = [
    path("", views.absenteeism_predict, name="absenteeism_predict"),

]
