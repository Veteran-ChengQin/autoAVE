# {
#   "model": "qwen3-vl-plus",
#   "messages": [
#     {
#       "role": "user",
#       "content": [
#         {
#           "type": "video_url",
#           "video_url": {
#             "url": "https://m.media-amazon.com/images/S/vse-vms-transcoding-artifact-us-east-1-prod/v2/3540a0b1-db1a-5e96-9437-53ad36e7cabb/ShortForm-Generic-480p-16-9-1409173089793-rpcbe5.mp4"
#           }
#         },
#         {
#           "type": "text",
#           "text": "该视频主要用于介绍假发，请你从视频中提取商品的这几个属性的值，以json格式输出，属性列表为:['Color', 'Hair Type', 'Material']"
#         }
#       ]
#     }
#   ],
#   "stream": true,
#   "stream_options": {
#     "include_usage": true
#   }
# }