import Button from './Components/Button'
import Header1 from './Components/Header1'
import Header2 from './Components/Header2'
import Input from './Components/Input'
import Form from './Components/Form'
import Label from "./Components/Label"
import FeatureList from './Components/FeatureList'

function App() {

    const handleScatter = () => {
        var rowValues = document.getElementById("pasteData").value.split("\t"); // Get the pasted row values and split them by tab

        var inputFields = document.querySelectorAll("#predictionForm");
        for (var i = 0; i < inputFields.length; i++) {
            inputFields[i].value = rowValues[i];
        }
    }
    const handleChurn = () => {
        fetch("http://localhost:8080/predict", {
            method: "POST",
            mode: "cors",
            body: JSON.stringify()
        })
    }

    return (
        <main className='main'>
            <div className='container'>
                <Header1 className={"display-1"} children={"Customer Churn Analysis"} />
                <Header2 className={"mt-5"} children={"Paste Excel row here."}></Header2>
                <div className='d-flex flex-row form-floating align-items-center'>
                    <Input id={"pasteData"}  className={"w-75 form-control d-flex"} type="text" placeholder="Paste Excel row here" />
                    <Label forLabel={"pasteData"} children={"Paste Excel row here"} />
                    <Button className={"d-flex ms-3 h-50 inline"} children={"Scatter Values"} onClick={handleScatter} />
                </div>
                <Header2 className={"mt-3"} children={"...Or input below each value individually"}></Header2>
                <Form id={"predictionForm"} className={"form-floating"} action={"/predict"} method={"post"}>
                    {/* <Input id={"state"} className={"form-control"} type={"text"} placeholder={"state"} required={"required"} />
                    <Label forLabel={"state"} children={"State"} /> */}
                    <FeatureList />
                    <Button className={"primary my-3"} children={"Predict Churn"} onClick={handleChurn} />
                </Form>
            </div>
        </main>
    )
}

export default App
