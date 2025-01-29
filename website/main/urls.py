from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('docs/', views.docs, name='docs'),
    path('contact/', views.contact, name='contact'),
]
