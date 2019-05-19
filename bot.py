from telegram.ext import Updater, Filters
from telegram.ext import CommandHandler, MessageHandler
import pickle

with open("model_persist.pkl", "rb") as f:
  model_download = pickle.load(f)
  
naive_bayes = model_download['model']
count_vector = model_download['vector']

updater = Updater(token="token da api")
dispatcher = updater.dispatcher

def start(bot, update):

    msg = "Olá eu sou o Santi, feito para você descarregar sua raiva. \n Descarregue sua raiva até produzir coisas positivas"

    bot.send_message(
        chat_id=update.message.chat_id,
        text=msg
    )

def echo(bot, update):
  msg = "Voce Digitou: " + update.message.text
  vetormsg = [msg]
  predicts = count_vector.transform(vetormsg).toarray()
  array = naive_bayes.predict(predicts)
  print(array)
  
  if([0]in array):
    bot.send_message(chat_id=update.message.chat_id,text="Você pode melhorar esses pensamentos para algo positivo, meu lider")
    print(0)
  else:
    bot.send_message(chat_id=update.message.chat_id,text="Estou gostando de ver que você está melhorando")
    print(1)


start_handler = CommandHandler('start', start)
echo_handler = MessageHandler(Filters.text, echo)
inferno_handler = MessageHandler(Filters.text, echo)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(echo_handler)

updater.start_polling()
updater.idle()