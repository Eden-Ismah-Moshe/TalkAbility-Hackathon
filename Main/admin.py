from django.contrib import admin
from .models import Customer, Request, Representative

# Register your models here.
admin.site.register(Customer)
admin.site.register(Request)
admin.site.register(Representative)
