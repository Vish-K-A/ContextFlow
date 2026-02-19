from django.urls import path
from . import views  

urlpatterns = [
    path("", views.ChatUploadView.as_view(), name="documents"),   
    path("files/", views.user_file_views, name="files"),
    path("chats/", views.chats_history, name="chats"),
    path("chat/followup/", views.chat_followup, name="chat_followup"),
    path("files/delete/<int:file_id>/", views.delete_file, name="delete_file"),
    path("conversation/<int:conversation_id>/", views.load_conversation, name="load_conversation"),
    path("conversation/<int:conversation_id>/delete/", views.delete_conversation, name="delete_conversation"),
    path("conversation/new/", views.new_conversation, name="new_conversation"),
    path("files/load/<int:file_id>/", views.load_file_in_chat, name="load_file_in_chat"),
]