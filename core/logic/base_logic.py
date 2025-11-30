from core.blackboard import Blackboard
from core.input_driver import InputDriver

class BaseLogic:
    def __init__(self, blackboard: Blackboard, input_driver: InputDriver):
        self.bb = blackboard
        self.input = input_driver
        
    def execute(self, role: str) -> bool:
        """
        执行逻辑
        :param role: 角色标识符 (e.g. 'leader')
        :return: True if action was taken, False otherwise
        """
        raise NotImplementedError
