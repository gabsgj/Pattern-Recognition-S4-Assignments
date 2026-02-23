from flask import Flask,request,jsonify,render_template
import numpy as np

app=Flask(__name__)

class HMM:
    def __init__(self,n_states,n_obs):
        self.N=n_states
        self.M=n_obs
        self.A=np.random.rand(self.N,self.N)
        self.A/=self.A.sum(axis=1,keepdims=True)
        self.B=np.random.rand(self.N,self.M)
        self.B/=self.B.sum(axis=1,keepdims=True)
        self.pi=np.random.rand(self.N)
        self.pi/=self.pi.sum()

    def forward(self,O):
        T=len(O)
        alpha=np.zeros((T,self.N))
        alpha[0]=self.pi*self.B[:,O[0]]
        for t in range(1,T):
            for j in range(self.N):
                alpha[t,j]=np.sum(alpha[t-1]*self.A[:,j])*self.B[j,O[t]]
        return alpha

    def backward(self,O):
        T=len(O)
        beta=np.zeros((T,self.N))
        beta[-1]=1
        for t in range(T-2,-1,-1):
            for i in range(self.N):
                beta[t,i]=np.sum(self.A[i]*self.B[:,O[t+1]]*beta[t+1])
        return beta


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train",methods=["POST"])
def train():

    data=request.json
    O=list(map(int,data["sequence"].split()))
    n_states=int(data["states"])
    n_obs=int(data["symbols"])
    iterations=int(data["iterations"])

    hmm=HMM(n_states,n_obs)

    history=[]
    prev=None
    tol=float(data.get("tolerance",1e-4))
    patience=int(data.get("patience",2))
    stable_steps=0
    converged=False

    for it in range(iterations):

        alpha=hmm.forward(O)
        beta=hmm.backward(O)

        P=float(np.sum(alpha[-1]))
        logL=float(np.log(P+1e-12))
        negLL=float(-logL)

        delta=None if prev is None else logL-prev
        prev=logL

        gamma=(alpha*beta)/P
        xi=np.zeros((len(O)-1,n_states,n_states))

        for t in range(len(O)-1):
            denom=np.sum(alpha[t][:,None]*hmm.A*hmm.B[:,O[t+1]]*beta[t+1])
            xi[t]=(alpha[t][:,None]*hmm.A*hmm.B[:,O[t+1]]*beta[t+1])/denom

        hmm.pi=gamma[0]

        for i in range(n_states):
            for j in range(n_states):
                hmm.A[i,j]=np.sum(xi[:,i,j])/np.sum(gamma[:-1,i])

        for j in range(n_states):
            for k in range(n_obs):
                mask=(np.array(O)==k)
                hmm.B[j,k]=np.sum(gamma[mask,j])/np.sum(gamma[:,j])

        history.append({
        "iter":it+1,
        "P":P,
        "logL":logL,
        "delta":delta,
        "negLL":negLL,
        "A":hmm.A.tolist(),
        "B":hmm.B.tolist(),
        "pi":hmm.pi.tolist(),   # NEW
        "alpha":alpha.tolist() if it<20 else None,
        "beta":beta.tolist() if it<20 else None,
        "gamma":gamma.tolist() if it<20 else None
        })

        if delta is not None and abs(delta)<tol:
            stable_steps+=1
        else:
            stable_steps=0

        if stable_steps>=patience:
            converged=True
            break

    return jsonify({
        "history":history,
        "meta":{
            "requested_iterations":iterations,
            "executed_iterations":len(history),
            "converged":converged,
            "tolerance":tol,
            "patience":patience
        },
        "final":{
            "A":hmm.A.tolist(),
            "B":hmm.B.tolist(),
            "pi":hmm.pi.tolist()
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
