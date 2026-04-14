from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('demographics/', views.demographics_view, name='demographics'),
    path('medications/', views.medications_view, name='medications'),
]