import Button from '../components/Button'
import Input from '../components/Input'
import Label from "../components/Label"
import InputFeature from '../components/InputFeature'
import React from 'react'
import { useState, useEffect } from 'react'
import Header3 from '../components/Header3'
import Heroe from '../components/Heroe'
import ChurnPercentage from '../components/ChurnPercentage'

function SingleCustomer() {

    const [feature_list, setFeatureList] = useState([{}])
    useEffect(() => {
        fetch("/feature_names").then(
            res => res.json()
        ).then(
            data => {
                data.pop()
                setFeatureList(data)
            }
        )
    }, [])
    let [churn, setChurn] = useState([{}])

    const handleScatter = () => {
        var rowValues = document.getElementById("pasteData").value.split("\t"); // Get the pasted row values and split them by tab

        var inputFields = document.querySelectorAll("[type=number]");
        for (var i = 0; i < inputFields.length; i++) {
            inputFields[i].value = rowValues[i];
        }
    }
    const handleClear = () => {
        feature_list.forEach((feature) => {
            document.querySelector("#"+ feature.name).value = ""
        })
    }
    const handleChurn = () => {
        let feature_post = []
        let inputFields = document.querySelectorAll("[type=number]");
        for(let i = 0; i < inputFields.length; i++) {
            let name = inputFields[i].id.split(" ")[0]
            let value = inputFields[i].value
            feature_post.push({
                "name": name,
                "value": value
            })
        }
        console.log("feature_post: ", feature_post)

        fetch("http://localhost:8080/predict", {
            method: "POST",
            mode: "cors",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ feature_post })
        }).then(
            res => res.json()
        ).then(
            data => {
                console.log(data)
                setChurn(data)
            }
        ).catch(
            e => console.error("No prediction yet\n", e)
        )
        
    }

    return (
        <>
            <div className='container'>
                <div className='card bg-light'>
                    <Heroe
                        children={"Single Customer Churn"}
                        display={"display-1"}
                        subheading={"Predict a customer's churn!"}
                        scrollButton={"Enter Customer Data"} />
                    <div className='card-body'>
                        <div className='row'>
                            <div className='col-md-6 col-sm-9'>
                                <div className='row d-flex flex-row align-items-center pt-5'>
                                    <div className='d-flex my-auto'>
                                        <div className='col form-floating'>
                                            <Input id={"pasteData"} className={"form-control"} type="text" placeholder="Paste Excel row here" />
                                            <Label forLabel={"pasteData"} children={"Excel Row"} />
                                        </div>
                                        <Button theme={"primary"} className={"col-lg-2 col-sm-3 my-md-2 ms-3"} children={"Scatter Values"} onClick={handleScatter} />
                                    </div>
                                </div>
                                <div id={"predictionForm"} className={"form row mt-3 flex-sm-row flex-column"}>
                                    <Header3 className={""}>Customer Information</Header3>
                                    {feature_list.map((feature) => {
                                        return <div className='form-floating col-md-4 col-sm-10 my-md-1'>
                                            <InputFeature id={feature.name} className={"form-control d-flex"} key={feature.name} feature={feature.name} />
                                            <Label forLabel={feature.name} children={feature.name} />
                                        </div>
                                    })}
                                    <div className='d-flex flex-row'>
                                            <Button theme={"primary"} className={"my-3 mx-2 flex-fill"} children={"Predict Churn"} onClick={handleChurn} />
                                            <Button theme={"outline-danger"} className={"my-3 mx-2 flex-fill"} children={"Clear"} onClick={handleClear} />
                                    </div>
                                </div>
                            </div>
                            <div className='col-md-4 my-auto'>
                                <div className='text-center py-5 unselectable'>
                                    <ChurnPercentage churn={churn} />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}

export default SingleCustomer