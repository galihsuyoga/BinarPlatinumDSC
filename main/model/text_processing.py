__author__ = 'galihsuyoga'

from main.model import db


class Abusive (db.Model):
    __tablename__ = 'abusive'
    word = db.Column(db.String(63), primary_key=True, nullable=False, unique=True, index=True, default="")
    # kepikiran identifikasi word masuk golongan apa aja
    is_HS = db.Column(db.SmallInteger, default=0)
    is_Abusive = db.Column(db.SmallInteger, default=0)
    is_HS_Individual = db.Column(db.SmallInteger, default=0)
    is_HS_Group = db.Column(db.SmallInteger, default=0)
    is_HS_Religion = db.Column(db.SmallInteger, default=0)
    is_HS_Race = db.Column(db.SmallInteger, default=0)
    is_HS_Physical = db.Column(db.SmallInteger, default=0)
    is_HS_Gender = db.Column(db.SmallInteger, default=0)
    is_HS_Other = db.Column(db.SmallInteger, default=0)
    is_HS_Weak = db.Column(db.SmallInteger, default=0)
    is_HS_Moderate = db.Column(db.SmallInteger, default=0)
    is_HS_Strong = db.Column(db.SmallInteger, default=0)

    def __init__(self, words):
        self.word = words
        self.is_HS = 0
        self.is_Abusive = 0
        self.is_HS_Individual = 0
        self.is_HS_Group = 0
        self.is_HS_Religion = 0
        self.is_HS_Race = 0
        self.is_HS_Physical = 0
        self.is_HS_Gender = 0
        self.is_HS_Other = 0
        self.is_HS_Weak = 0
        self.is_HS_Moderate = 0
        self.is_HS_Strong = 0

    def __repr__(self):
        return '<Abusive %r>' % self.word

    def save(self):
        try:
            db.session.add(self)
            db.session.flush()
            db.session.commit()
        except:
            db.session.rollback()
            raise


class KamusAlay(db.Model):
    __tablename__ = 'kamus_alay'

    word = db.Column(db.String(63), primary_key=True, nullable=False, index=True, default="")
    meaning = db.Column(db.String(63), nullable=False, index=True, default="")

    Abusive = db.relationship("Abusive", primaryjoin="Abusive.word==foreign(KamusAlay.word)")

    def __init__(self, words, meaning):
        self.word = words
        self.meaning = meaning

    def __repr__(self):
        return '<KamusAlay %r>' % self.word

    def save(self):
        try:
            db.session.add(self)
            db.session.flush()
            db.session.commit()
        except:
            db.session.rollback()
            raise


class TextLog(db.Model):
    __tablename__ = 'text_log'

    id = db.Column(db.Integer().with_variant(db.Integer, "sqlite"), primary_key=True, nullable=False, index=True, autoincrement=True)
    raw_text = db.Column(db.String(255), nullable=False, index=True, default="")
    clean = db.Column(db.String(255), nullable=False, default="")

    def __init__(self, text, clean):
        self.raw_text = text
        self.clean = clean

    def __repr__(self):
        return '<TextLog %r>' % self.raw_text

    def save(self):
        try:
            db.session.add(self)
            db.session.flush()
            db.session.commit()
        except:
            db.session.rollback()
            raise

class AlayAbusiveLog(db.Model):
    __tablename__ = 'alay_abusive_log'

    id = db.Column(db.Integer().with_variant(db.Integer, "sqlite"), nullable=False, autoincrement=True)
    word = db.Column(db.String(31), primary_key=True, nullable=False, index=True, default="")
    clean = db.Column(db.String(31), primary_key=True, nullable=False, index=True, default="")
    foul_type = db.Column(db.String(255), nullable=False, default="")
    text_log_id = db.Column(db.Integer, nullable=False, index=True, default="")

    __foul_type_abusive = "ABUSE"
    __foul_type_alay = "ALAY"
    __foul_type_mixed = "MIXED"

    def __init__(self, word, clean, foul_type, log_id):
        self.word = word
        self.clean = clean
        self.foul_type = foul_type
        self.text_log_id = log_id

    def __repr__(self):
        return '<TextFileTweetLog %r>' % self.word

    def save(self):
        try:
            db.session.add(self)
            db.session.flush()
            db.session.commit()
        except:
            db.session.rollback()
            raise

    @classmethod
    def foul_type_abusive(cls):
        return cls.__foul_type_abusive

    @classmethod
    def foul_type_alay(cls):
        return cls.__foul_type_alay

    @classmethod
    def foul_type_mixed(cls):
        return cls.__foul_type_mixed

class FileTextLog(db.Model):
    __tablename__ = 'file_upload_text_log'

    ID = db.Column(db.Integer().with_variant(db.Integer, "sqlite"), primary_key=True, nullable=False, index=True, autoincrement=True)
    Tweet = db.Column(db.String(255), nullable=False, index=True, default="")
    Clean = db.Column(db.String(255), nullable=False, default="")
    HS = db.Column(db.SmallInteger, default=0)
    Abusive = db.Column(db.SmallInteger, default=0)
    HS_Individual = db.Column(db.SmallInteger, default=0)
    HS_Group = db.Column(db.SmallInteger, default=0)
    HS_Religion = db.Column(db.SmallInteger, default=0)
    HS_Race = db.Column(db.SmallInteger, default=0)
    HS_Physical = db.Column(db.SmallInteger, default=0)
    HS_Gender = db.Column(db.SmallInteger, default=0)
    HS_Other = db.Column(db.SmallInteger, default=0)
    HS_Weak = db.Column(db.SmallInteger, default=0)
    HS_Moderate = db.Column(db.SmallInteger, default=0)
    HS_Strong = db.Column(db.SmallInteger, default=0)

    def __init__(self, text, clean, full):
        self.Tweet = text
        self.Clean = clean
        self.HS = int(full['HS']) if 'HS' in full else 0

        self.Abusive = int(full['Abusive']) if 'Abusive' in full else 0

        self.HS_Individual = int(full['HS_Individual']) if 'HS_Individual' in full else 0

        self.HS_Group = int(full['HS_Group']) if 'HS_Group' in full else 0
        self.HS_Religion = int(full['HS_Religion']) if 'HS_Religion' in full else 0
        self.HS_Race = int(full['HS_Race']) if 'HS_Race' in full else 0
        self.HS_Physical = int(full['HS_Physical']) if 'HS_Physical' in full else 0
        self.HS_Gender = int(full['HS_Gender']) if 'HS_Gender' in full else 0
        self.HS_Other = int(full['HS_Other']) if 'HS_Other' in full else 0
        self.HS_Weak = int(full['HS_Weak']) if 'HS_Weak' in full else 0
        self.HS_Moderate = int(full['HS_Moderate']) if 'HS_Moderate' in full else 0
        self.HS_Strong = int(full['HS_Strong']) if 'HS_Strong' in full else 0

    def __repr__(self):
        return '<FileTextLog %r>' % self.Tweet

    def save(self):
        try:
            db.session.add(self)
            db.session.flush()
            db.session.commit()
        except:
            db.session.rollback()
            raise



class AlayAbusiveFileLog(db.Model):
    __tablename__ = 'alay_abusive_file_log'

    id = db.Column(db.Integer().with_variant(db.Integer, "sqlite"), nullable=False, autoincrement=True)
    word = db.Column(db.String(31), primary_key=True, nullable=False, index=True, default="")
    clean = db.Column(db.String(31), primary_key=True, nullable=False, index=True, default="")
    foul_type = db.Column(db.String(255), nullable=False, default="")
    file_upload_text_log_id = db.Column(db.Integer, nullable=False, index=True, default="")

    __foul_type_abusive = "ABUSE"
    __foul_type_alay = "ALAY"
    __foul_type_mixed = "MIXED"

    def __init__(self, word, clean, foul_type, log_id):
        self.word = word
        self.clean = clean
        self.foul_type = foul_type
        self.file_upload_text_log_id = log_id

    def __repr__(self):
        return '<AlayAbusiveFileLog %r>' % self.word

    def save(self):
        try:
            db.session.add(self)
            db.session.flush()
            db.session.commit()
        except:
            db.session.rollback()
            raise

    @classmethod
    def foul_type_abusive(cls):
        return cls.__foul_type_abusive

    @classmethod
    def foul_type_alay(cls):
        return cls.__foul_type_alay

    @classmethod
    def foul_type_mixed(cls):
        return cls.__foul_type_mixed

class RawText(db.Model):
    __tablename__ = 'raw_text'

    id = db.Column(db.Integer().with_variant(db.Integer, "sqlite"), nullable=False, autoincrement=True)
    kalimat = db.Column(db.String(512), primary_key=True, nullable=False, index=True, default="")
    sentimen = db.Column(db.String(31), primary_key=True, nullable=False, index=True, default="")

    __sentimen_type_netral = "netral"
    __sentimen_type_positif = "positif"
    __sentimen_type_negatif = "negatif"

    def __init__(self, kalimat, sentimen):
        self.kalimat = kalimat
        self.sentimen = sentimen


    def __repr__(self):
        return '<RawText %r>' % self.kalimat

    def save(self):
        try:
            db.session.add(self)
            db.session.flush()
            db.session.commit()
        except:
            db.session.rollback()
            raise


class ProcessedText(db.Model):
    __tablename__ = 'processed_text'

    id = db.Column(db.Integer().with_variant(db.Integer, "sqlite"), nullable=False, autoincrement=True)
    kalimat = db.Column(db.String(512), primary_key=True, nullable=False, index=True, default="")
    sentimen = db.Column(db.String(31), primary_key=True, nullable=False, index=True, default="")

    kalimat_bersih = db.Column(db.String(512), nullable=False, index=True, default="")
    kalimat_bersih_v2 = db.Column(db.String(512), nullable=False, index=True, default="")
    kalimat_bersih_v3 = db.Column(db.String(512), nullable=False, index=True, default="")
    jumlah_kalimat = db.Column(db.Integer, nullable=False, default="")

    jumlah_kata = db.Column(db.Integer, nullable=False, default="")



    def __init__(self, kalimat, sentimen, kalimat_bersih, kalimat_bersih_v2, kalimat_bersih_v3, jumlah_kalimat, jumlah_kata):
        self.kalimat = kalimat
        self.sentimen = sentimen
        self.kalimat_bersih = kalimat_bersih
        self.kalimat_bersih_v2 = kalimat_bersih_v2
        self.kalimat_bersih_v3 = kalimat_bersih_v3
        self.jumlah_kalimat = jumlah_kalimat
        self.jumlah_kata = jumlah_kata


    def __repr__(self):
        return '<ProcessedText %r>' % self.kalimat_bersih_v3

    def save(self):
        try:
            db.session.add(self)
            db.session.flush()
            db.session.commit()
        except:
            db.session.rollback()
            raise