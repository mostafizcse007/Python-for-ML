class Account:
    bank_name = "Phitron Bank"

    def __init__(self,name,acc_no,balance):
        self.name = name
        self.acc_no = acc_no
        self.__balance = balance

    def get_balance(self):
        return self.__balance

    def set_balance(self,bal):
        self.__balance = bal

    def deposit(self, money):
        self.__balance += money
        return (f"{money} is added."
                f"New balance {self.__balance}")

    def withdraw(self,money):
        if money> self.__balance:
            return f"Insufficient Balance"
        self.__balance -= money
        return f"{money} withdrawed successfully"

class SavingAccount(Account):
    def __init__(self, name, acc_no, balance,interest_rate):
        super().__init__(name, acc_no, balance)
        self.interest_rate = interest_rate

    def calculate_interest_rate(self):
        interest = self.get_balance() * self.interest_rate
        return f"Interest rate is {interest}"

class CurrentAccount(Account):

    def __init__(self, name, acc_no, balance,overlimit):
        super().__init__(name, acc_no, balance)
        self.overlimit = overlimit

    def withdraw(self,money):
        if money > (self.get_balance() + self.overlimit):
            return f"Insufficient Balance"
        new_bal = self.get_balance() - money
        self.set_balance(new_bal)
        return f"{money} withdrawed successfully"



currentUser = CurrentAccount('Antu',123,1000,5000)
print(currentUser.get_balance())
print(currentUser.withdraw(100000))



