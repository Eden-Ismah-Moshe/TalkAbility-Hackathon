from django.contrib import admin
from .models import Customer, Request, Agent

# Register your models here.
admin.site.register(Customer)
admin.site.register(Request)
admin.site.register(Agent)
