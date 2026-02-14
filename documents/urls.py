from django.urls import path
from . import views  

urlpatterns=[
    path("",views.ChatUploadView.as_view(),name="documents"),   
    path("files/",views.user_file_views,name="files"),
]