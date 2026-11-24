from datetime import datetime
import random
import os
import numpy as np
import torch
from models import ValueNet

shortToLongPath = "../resource/shorttolong"
predicatesEncodeDictPath = "./predicatesEncodedDict"


class data:
    def __init__(self, state, time):
        self.state = state
        self.time = time
        self.label = 0


class supervised:
    def __init__(self, args):
        # Read dict predicatesEncoded
        f = open(predicatesEncodeDictPath, "r")
        a = f.read()
        self.predicatesEncodeDict = eval(a)
        f.close()

        # Read all tablenames and get tablename-number mapping
        tables = []
        f = open(shortToLongPath, "r")
        a = f.read()
        short_to_long = eval(a)
        f.close()
        for i in short_to_long.keys():
            tables.append(i)
        tables.sort()
        self.table_to_int = {}
        for i in range(len(tables)):
            self.table_to_int[tables[i]] = i

        # The dimension of the network input vector
        self.num_inputs = len(tables) * len(tables) + len(
            self.predicatesEncodeDict["1a"]
        )
        # The dimension of the vector output by the network
        self.num_output = 5
        self.args = args
        # How to interpret timeouts when building the dataset
        self.timeout_value = int(getattr(args, "timeout_value", 10**9))
        self.right = 0

        # build up the network
        self.value_net = ValueNet(self.num_inputs, self.num_output)
        # check some dir
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.dataList = []
        self.testList = []

    # Parsing query plan
    def hint2matrix(self, hint):
        tablesInQuery = hint.split(" ")
        # NumPy 2.0: np.mat was removed, use a regular ndarray
        matrix = np.zeros((len(self.table_to_int), len(self.table_to_int)))
        stack = []
        difference = 0
        for i in tablesInQuery:
            if i == ")":
                tempb = stack.pop()
                tempa = stack.pop()
                _ = stack.pop()
                b = tempb.split("+")
                a = tempa.split("+")
                b.sort()
                a.sort()
                indexb = self.table_to_int[b[0]]
                indexa = self.table_to_int[a[0]]
                matrix[indexa, indexb] = (len(tablesInQuery) + 2) / 3 - difference
                difference += 1
                stack.append(tempa + "+" + tempb)
                # print(stack)
            else:
                stack.append(i)
        return matrix

    # Divide training set and test set
    def pretreatment(self, path):
        # Ensure output directory exists (for the future, if we ever want to write files)
        if not os.path.exists("./data"):
            os.makedirs("./data", exist_ok=True)
        # Load data uniformly and randomly select for training
        file_test = open(path)
        line = file_test.readline()
        while line:
            queryName = line.split(",")[0].encode("utf-8").decode("utf-8-sig").strip()
            hint = line.split(",")[1]
            matrix = self.hint2matrix(hint)
            predicatesEncode = self.predicatesEncodeDict[queryName]
            # matrix.flatten().tolist() already returns a flat list of numbers
            state = matrix.flatten().tolist()
            state = state + predicatesEncode
            runtime_str = line.split(",")[2].strip()
            try:
                runtime = int(float(runtime_str))
            except Exception:
                # set statement timeout
                runtime = self.timeout_value
            temp = data(state, runtime)
            self.dataList.append(temp)
            line = file_test.readline()

        self.dataList.sort(key=lambda x: x.time, reverse=False)
        for i in range(self.dataList.__len__()):
            self.dataList[i].label = int(
                i / (self.dataList.__len__() / self.num_output + 1)
            )
            # print(self.dataList[i].label)
        for i in range(int(self.dataList.__len__() * 0.3)):
            index = random.randint(0, len(self.dataList) - 1)
            temp = self.dataList.pop(index)
            self.testList.append(temp)

        print(
            "size of test set:",
            len(self.testList),
            "\tsize of train set:",
            len(self.dataList),
        )
        # Previously, the dataset was serialized to files using pickle.
        # In Python 3 with our proxied module scheme this causes errors.
        # For training and testing it's sufficient to keep the data in memory.

    # functions to train the network
    def supervised(self):
        self.load_data()
        optim = torch.optim.SGD(self.value_net.parameters(), lr=0.01)
        # loss_func = torch.nn.MSELoss()
        # loss_func = torch.nn.CrossEntropyLoss()
        loss_func = torch.nn.NLLLoss()
        loss1000 = 0
        count = 0

        for step in range(1, 600001):
            index = random.randint(0, len(self.dataList) - 1)
            state = self.dataList[index].state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            predictionRuntime = torch.log(self.value_net(state_tensor) + 1e-10)
            predictionRuntime = predictionRuntime.view(1, -1)

            label = []
            label.append(self.dataList[index].label)
            label_tensor = torch.tensor(label)

            loss = loss_func(predictionRuntime, label_tensor)

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss1000 += loss.item()
            if step % 1000 == 0:
                print(
                    "[{}]  Epoch: {}, Loss: {:.5f}".format(
                        datetime.now(), step, loss1000
                    )
                )
                loss1000 = 0
                # Run validation and check for improvement
                self.test_network()
                print(
                    "[{}]  Epoch: {}, Loss: {:.5f}".format(
                        datetime.now(), step, loss1000
                    )
                )
            if step % 200000 == 0:
                torch.save(
                    self.value_net.state_dict(), self.args.save_dir + "supervised.pt"
                )
                self.test_network()

    # functions to test the network
    def test_network(self):
        self.load_data()
        model_path = self.args.save_dir + "supervised.pt"
        # In the original code this used actor_net, but in this version
        # we only have value_net — so we also use it for testing.
        # On early iterations the model file may not exist yet.
        if os.path.exists(model_path):
            self.value_net.load_state_dict(
                torch.load(model_path, map_location=lambda storage, loc: storage)
            )
        self.value_net.eval()

        correct = 0
        for step in range(self.testList.__len__()):
            state = self.testList[step].state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            predictionRuntime = self.value_net(state_tensor)
            prediction = predictionRuntime.detach().cpu().numpy()
            maxindex = np.argmax(prediction)
            label = self.testList[step].label
            # print(maxindex, "\t", label)
            if maxindex == label:
                correct += 1
        print(
            correct, self.testList.__len__(), correct / self.testList.__len__(), end=" "
        )

        correct1 = 0
        for step in range(self.dataList.__len__()):
            state = self.dataList[step].state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            predictionRuntime = self.value_net(state_tensor)
            # prediction = predictionRuntime.detach().cpu().numpy()[0]
            prediction = predictionRuntime.detach().cpu().numpy()
            maxindex = np.argmax(prediction)
            label = self.dataList[step].label
            # print(maxindex, "\t", label)
            if maxindex == label:
                correct1 += 1
        print(correct1, self.dataList.__len__(), correct1 / self.dataList.__len__())
        self.right = correct / self.testList.__len__()

    def load_data(self):
        """
        In the simplified flow the data is already loaded by pretreatment()
        before training/testing. If not, we load it from the original CSV.
        """
        if self.dataList:
            return
        # If the arrays are empty, run pretreatment again
        if hasattr(self.args, "data_file"):
            self.pretreatment(self.args.data_file)
        else:
            raise RuntimeError("No data loaded and no args.data_file specified")
