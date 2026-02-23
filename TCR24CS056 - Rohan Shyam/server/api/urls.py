
from django.urls import path
from . import views

urlpatterns = [
    path('run/', views.run_baum_welch, name='run-baum-welch'),
]