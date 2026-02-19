from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth.models import User


class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=200, default="New conversation") 
    created_at = models.DateTimeField(auto_now_add=True)
    last_pdf_path = models.CharField(max_length=500, blank=True)
    
    class Meta:
        ordering = ['-created_at']  
    
    def __str__(self):
        return f"{self.title} - {self.user.username if self.user else 'Anonymous'}"


class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, null=True, blank=True, related_name='messages')
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    text = models.TextField(blank=True)
    is_ai_response = models.BooleanField(default=False) 
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Message {self.id} by {self.user.username if self.user else 'AI'}"


class Attachment(models.Model):
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name='attachments')
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    file = models.FileField(
        upload_to='uploads/pdfs/%Y/%m/%d/',   
        validators=[FileExtensionValidator(allowed_extensions=['pdf'])]  
    )
    uploaded_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    file_size = models.BigIntegerField(default=0)
    description = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.file.name}"
    
    def save(self, *args, **kwargs):
        if self.file:
            self.file_size = self.file.size
        super().save(*args, **kwargs)