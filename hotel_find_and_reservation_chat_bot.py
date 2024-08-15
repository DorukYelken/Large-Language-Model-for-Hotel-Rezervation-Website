# Author: Muhammet Eryılmaz
# Author: Doruk Yelken

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, Namespace, fields
import time
from openai import OpenAI
import json

import numpy as np
from pymongo.mongo_client import MongoClient

app = Flask(__name__)
api = Api(app)
ns = Namespace("api", description="Sample APIs")

chat_model = ns.model('Chat', {
    'prompt': fields.String(required=True, description='User prompt') , 'conv_id': fields.String(required=True, description='User id')
})

#Kullanıcı rezervasyon bilgililerini doldurduğunda kullanıcıya ask_user fonksiyonuyla bu rezervasyonu veya rezervasyon iptalinin onayını kullanıcıya sor.
client = OpenAI()
GPT_MODEL = "gpt-4-turbo"          
history_obj = {}
rez_obj = {}

#history = []        Kullanıcı rezervasyon bilgililerini doldurduğunda kullanıcıya ask_user fonksiyonuyla bu rezervasyonu veya rezervasyon iptalinin onayını kullanıcıya sor.
history_res= []
history_rec = []

dosya_yolu = 'HotelsDatabase.txt'
with open(dosya_yolu, 'r') as dosya:
    icerik = dosya.read()
historyI = [
        {"role": "system", "content": "**[MOST IMPORTANT always use functions in tools to give response]** Sen bir otel rezervasyon sitesinin Main agentısın. Senin işin user ve diğer assistanların sana yazdığı istekleri anlayıp ilgili fonksiyonlara yönlendirmek, kendin kullanıcıdan bigi isteme diğer agentlar o işi yapacak zaten, sen sadece fonksiyonları çalıştır, sadece gereken durumlarda ask_user fonksiyonunu kullanıp user ile iletişime geç. Görevlerin: -User rezervasyon yapmak istiyorsa reservation_agent_func'ı çalıştır. -User rezervasyon iptal etmek istiyorsa reservation_agent_func'ı çalıştır.  - Müşteri otel aradığını ya da belirli bir otel ile ilgili sorusunu belirtirse hotel_recommender_agent_func'ı çalıştır. Ama müşteriye otellerin önerildiği ya da daha fazla bilgi talep edildiği bir assistant jsonı gelirse bunu ask_user fonksiyonu ile kullanıcıya sor. Eğer assistant otel ismini kaydettiyse reservation_agent_func fonksiyonunu çalıştır. kullanıcının rezervasyon bilgilerinin olduğu bir json geldiğinde ask_user kullanarak bilgilerin doğruluğunu onaylamasını iste. "}
]
history_resI = [
        {"role": "system", "content": "Sen reservationAgent isminde bir otel rezervasyon sitesinin rezervasyondan sorumlu agentısın. Diğer fonksiyonları kullanmıyorsan return_response fonksiyonunu kullan. Her zaman fonksiyonla cevap ver. Text şeklide konuşma.-reservasyon yapmak için make_reservation fonksiyonunu kullan ve bu fonksiyonun tamamını doldurana kadar kullanıcıdan return_response fonksiyonuyla bilgileri iste, bu bilgiler: `name`, `surname`, `hotel_name`, `date_in`, `date_out`, `city`, `for_how_many_people'. -rezervasyon iptal etmek için cancel_reservation fonksiyonunu kullan ve bu fonksiyonun tamamını doldurana kadar kullanıcıdan return_response fonksiyonuyla bilgileri iste, bu bilgiler: `name`, `surname`, `hotel_name`, `date_in`, `date_out`, `city`, `for_how_many_people'. Eğer tüm bilgiler girilmediyse eksik olan bilgileri return_response fonksiyonuyla kullanıcıdan iste. Rezervasyon bilgilerinin tamamını aldıktan sonra kullanıcıdan onay al ve bu işlemleri rezervasyon yapmak içinse reservation_check_user; rezervasyon iptal etmek içinse cancel_check_user fonksiyonunu döndür Eğer kullanıcı onayla ilgili bişey belirtmediyse tekrar sor. IMPORTANT[**Onay AL] make_reservation fonksiyonu tam dolduğunda  return_response kullanarak bilgilerin doğruluğunu onaylamasını iste."}
]
history_recI = [
        {"role": "system", "content": f"Sen bir otel rezervasyon sitesinin müşteriye istediği oteli bulmasında yardımcı olacak agentısın. Sitede bir tane main agent bir tane reservation agent ve bir de sen varsın. Main agent müşteri ile iletişimden sorumlu, reservation agent müşterinin rezervasyonundan sorumlu, sen de müşterinin hangi otelde konaklayacığından ve bunu seçmesinden sorumlusun. return_response fonksiyonunu kullanarak müşteriye uygun otelleri listele. düz text şeklide konuşamazsın. - Her zaman cevabını return_response fonksiyonuna sok!. müşteriden ekstra bilgi almak istersen, sonucu onaylatmak istersen ki otel ismi gelirse onaylatmak için sor ya da sonucu return etmek istersen return_response fonksiyonunu kullan. müşteri  otel isterse Otelleri bu database den oner:{icerik} müşteri otelini seçtiğinde find_hotel fonksiyonu ile otel ismini kaydet."}
]

main_tools = [
    {
        "type": "function",
        "function": {
            "name": "reservation_agent_func",
            "description": "Take user Query about user reservation request reservetion information text input in JSON format and then send this request in JSON format to the reservationAgent",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "user reservation request"
                    }
                },
                "required": ["response"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask user for more information",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "taken response from user"
                    }
                },
                "required": ["response"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "hotel_recommender_agent_func",
            "description": "when customer ask hotel or somethings about hotel you will take this request and rotate to hotelRecommenderAgent in JSON format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "fill with user request"
                    }
                },
                "required": [
                "response"
                ]
            }
        } 
    }
]

recommender_tools = [
    {
        "type":"function",
        "function": {
            "name": "find_hotel",
            "description": "choose the available hotel for customer then fill the relevant part with this. if there is missing information, write in there like 'There is a missing information!!' ",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotel_name": {
                        "type": "string",
                        "description": "fill with hotel name"
                    },
                    
                    "city": {
                        "type": "string",
                        "description": "fill with city of the hotel"
                    }

                },
                "required": [
                "hotel_name" , "city"
                ]
            }
        }
    },

    {
        "type":"function",
        "function": {
            "name": "return_response",
            "description": "return a response on JSON format that have hotel information or requiered informations request",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "response that has hotel information or additional things"
                    }
                },
                "required": [
                "response"
                ]
            }
        }
    }
]

reservation_tools = [
    {
        "type":"function",
        "function": {
            "name": "make_reservation",
            "description": "fill the relevant fields according to given informations and return json format. if there is missing information, write in there like 'There is a missing information!!' ",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "fill with Customer name"
                    },
                    "surname": {
                        "type": "string",
                        "description": "fill with Customer surname"
                    },
                    "date_in": {
                        "type": "string",
                        "description": "fill with begin date of reservation"
                    },
                    "date_out": {
                        "type": "string",
                        "description": "fill with last date of reservation"
                    },
                    "hotel_name": {
                        "type": "string",
                        "description": "fill with hotel name"
                    },
                    "city": {
                        "type": "string",
                        "description": "fill with city where hotel located. find it from history not from user."
                    },
                    "for_how_many_people": {
                        "type": "string",
                        "description": "This reservation for how many people?"
                    }
                },
                "required": [
                    "name",
                    "surname",
                    "date_in",
                    "date_out",
                    "hotel_name",
                    "city",
                    "for_how_many_people"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "cancel_reservation",
            "description": "fill the relevant fields according to given informations and return json format",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "fill with Customer name"
                    },
                    "surname": {
                        "type": "string",
                        "description": "fill with Customer surname"
                    },
                    "date_in": {
                        "type": "string",
                        "description": "fill with begin date of reservation"
                    },
                    "date_out": {
                        "type": "string",
                        "description": "fill with last date of reservation"
                    },
                    "hotel_name": {
                        "type": "string",
                        "description": "fill with hotel name"
                    },
                    "city": {
                        "type": "string",
                        "description": "fill with city where hotel located"
                    },
                    "for_how_many_people": {
                        "type": "string",
                        "description": "This reservation for how many people?"
                    }
                },
                "required": [
                    "name",
                    "surname",
                    "date_in",
                    "date_out",
                    "hotel_name",
                    "city",
                    "for_how_many_people"
                ]
            }
        } 
    },

    {
        "type": "function",
        "function": {
            "name": "return_response",
            "description": "return a response on JSON format that have reservation informations or requiered informations request",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "response that has reservation informations or additional things"
                    }
                },
                "required": [
                    "response"
                ]
            }
        }  
    },
    {
    "type": "function",
    "function": {
        "name": "rezervation_check_user",
        "description": "check is user ok with givin hotel. If there is any approve or disapprove use this function.",
        "parameters": {
            "type": "object",
            "properties": {
            "check": {
                "type": "boolean",
                "description": "if user ok return ture else return false. "
            }
            },
            "required": [
            "check"
            ]
        }
    }
    },
        {
    "type": "function",
    "function": {
        "name": "cancelation_check_user",
        "description": "check is user ok to cancel this rezervation. If there is any approve or disapprove use this function.",
        "parameters": {
            "type": "object",
            "properties": {
            "check": {
                "type": "boolean",
                "description": "if user want to cancel return ture else return false. "
            }
            },
            "required": [
            "check"
            ]
        }
    }
    }    
]

def chat_completion_request(system ,messages, tools=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages= system + messages,
            tools=tools,
            tool_choice="required",
            temperature=0.5
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def update_history(item, unique, main):
    unique.append({"role":"assistant", "content": item})
    main.append({"role":"assistant", "content": item})

def get_function_name(chat):
    chat_function_name = chat.choices[0].message.tool_calls[0].function.name
    return chat_function_name

def get_function_arguments(chat):
    chat_function_arguments = chat.choices[0].message.tool_calls[0].function.arguments
    return chat_function_arguments

def main(conv_id, history, function_name_main , function_arguments_main , rez): 
    if function_name_main == "ask_user":
        res = json.loads(function_arguments_main)
        return jsonify(res)
    if function_name_main == "reservation_agent_func":
        history_res.append(history[-2])
        history_res.append(history[-1])
        reservation_chat = chat_completion_request( history_resI, history_res, tools=reservation_tools, model=GPT_MODEL)
        print(reservation_chat)

        function_name = get_function_name(reservation_chat)
        function_arguments = get_function_arguments(reservation_chat)

        output = json.loads(function_arguments)

        if function_name == "make_reservation" or function_name == "cancel_reservation":
            update_history(str(output), history_res, history) 
            rez[0] = output
            print("rez::::::::::: ", rez[0])       
            pass      
        if function_name == "return_response":
            update_history(output["response"], history_res, history)
            return output                 
        if function_name == "rezervation_check_user":
            print(str(output["check"]))
            if(str(output["check"]) == "True"):
                return jsonify(":::::::::::::::::::::::::::Rezervasyon gerçekleşti::::::::::::::::::::::::::::::" , "Rezervasyon Bilgileri: " , rez[0])
        if function_name == "cancelation_check_user":
            print(str(output["check"]))
            if(str(output["check"]) == "True"):
                return jsonify(":::::::::::::::::::::::::::Rezervasyon İptal edildi::::::::::::::::::::::::::::::" , "İptal Edilen Rezervasyon: " , rez[0])                
            
    if function_name_main == "hotel_recommender_agent_func":
        history_rec.append(history[-1])
        recommender_chat = chat_completion_request(history_recI, history_rec, tools=recommender_tools, model=GPT_MODEL)

        function_name = get_function_name(recommender_chat)
        function_arguments = get_function_arguments(recommender_chat)

        output = json.loads(function_arguments)

        if function_name == "find_hotel":
            update_history(str(output["hotel_name"] + " " + output["city"]) , history_rec, history)
            print("History::::  " , history)
            pass
        if function_name == "return_response":
            update_history(output["response"], history_rec, history)
            return output
        
    history_obj[conv_id] = history 
    """
    main_chat = chat_completion_request(historyI , history, tools=main_tools, model=GPT_MODEL)
    function_name_main = get_function_name(main_chat)  
    function_arguments_main = get_function_arguments(main_chat)   
    """ 
    return main(conv_id , history , function_name_main, function_arguments_main, rez)

def fill_object(c_id, obj):
    if not c_id in obj:
        obj[c_id] = []
    obj_value = obj[c_id]
    if not obj_value:
        obj_value=[]
    return obj_value

@ns.route("/chat")
class Chat(Resource):
    @ns.expect(chat_model)
    def post(self):
        data = request.json
        userPrompt = data['prompt']
        conv_id = data['conv_id']

        if not conv_id:
            raise Exception("Coversation id is required.")

        print(conv_id)

        history = fill_object(conv_id, history_obj)
        rez = fill_object(conv_id, rez_obj)
        history.append({"role":"user", "content":userPrompt})
        print("Rez Flask: " , rez)
        rez.append(" ")
        
        main_chat = chat_completion_request(historyI , history, tools=main_tools, model=GPT_MODEL)
        print(main_chat)
        function_name_main = get_function_name(main_chat)
        function_arguments = get_function_arguments(main_chat)
        output_main = json.loads(function_arguments)
        print(output_main)
        history.append({"role":"assistant", "content": output_main["response"]})
        print("HİSTORY:::: " , history)
        print("HİSTORY RES:::: " , history_res)
        history_obj[conv_id] = history
        print("history_obj:::::   " , history_obj)
        return main(conv_id , history , function_name_main , function_arguments, rez)
"""       
@ns.route("/get_history")
class Chat(Resource):
    @ns.expect()
    def get(self):
        return history
"""


api.add_namespace(ns, '/api')

if __name__ == '__main__':
    app.run(debug=True)


# Author: Muhammet Eryılmaz
# Author: Doruk Yelken