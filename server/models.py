from . import db

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    state = db.Column(db.IntegerInteger)
    account_length = db.Column(db.Integer)
    area_code = db.Column(db.Integer)
    international_plan = db.Column(db.Integer)
    voice_mail_plan = db.Column(db.Integer)
    number_vmail_messages = db.Column(db.Integer)
    total_day_minutes = db.Column(db.Integer)
    total_day_calls = db.Column(db.Integer)
    total_day_charge = db.Column(db.Integer)
    total_eve_minutes = db.Column(db.Integer)
    total_eve_calls = db.Column(db.Integer)
    total_eve_charge = db.Column(db.Integer)
    total_night_minutes = db.Column(db.Integer)
    total_night_calls = db.Column(db.Integer)
    total_night_charge = db.Column(db.Integer)
    total_intl_minutes = db.Column(db.Integer)
    total_intl_calls = db.Column(db.Integer)
    total_intl_charge = db.Column(db.Integer)
    number_customer_service_calls = db.Column(db.Integer)
    