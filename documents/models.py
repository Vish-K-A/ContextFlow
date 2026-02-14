from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth.models import User


# Create your models here.
class Message(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE,null=True,blank=True)
    text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Message {self.id} by {self.user.username if self.user else 'Anonymous'}"

    

class Attachment(models.Model):
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name='attachments')
    user = models.ForeignKey(User,on_delete=models.CASCADE,null=True,blank=True)
    file = models.FileField(upload_to='uploads/%Y/%m/%d/', validators=[FileExtensionValidator(allowed_extensions=['pdf', 'png', 'jpg', 'gif', 'webp','doc', 'docx', 'txt', 'md','xls', 'xlsx', 'csv','mp4', 'mov', 'avi' ])])
    uploaded_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    file_size = models.BigIntegerField(default=0)  
    file_type = models.CharField(max_length=100, blank=True)  
    description = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.file.name}"
    
    def save(self, *args, **kwargs):
        if self.file:
            self.file_size = self.file.size
            self.file_type = self.file.name.split('.')[-1].lower()
        super().save(*args, **kwargs)