"""
This class is used to return either success
or failure response to the user
"""
class Response:

    @classmethod
    def success(cls, data):
        response = dict()
        response['success'] = True
        response['data'] = data
        return response

    @classmethod
    def failure(cls, message):
        response = dict()
        response['success'] = False
        response['message'] = message
        return response
