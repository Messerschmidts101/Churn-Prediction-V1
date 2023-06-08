import React, { useState, useEffect } from 'react'
import Button from './Components/Button'
import Header1 from './Components/Header1'
import Header2 from './Components/Header2'
import Input from './Components/Input'
import Form from './Components/Form'
import Label from "./Components/Label"
import FeatureList from './Components/FeatureList'

function App() {

    const [data, setData] = useState([{}])

    useEffect(() => {
        fetch("/members").then(
            res => res.json()
        ).then(
            data => {
                setData(data)
                console.log(data)
            }
        )
    }, [])
    
    const feature_names = [
        {
            id: 0,
            name:"state"
        },
        {
            id: 1,
            name: "account_length"
        },
        {
            id: 2,
            name: "area_code"
        },
        {
            id: 3,
            name: "international_plan"
        },
        {
            id: 4,
            name: "voice_mail_plan"
        },
        {
            id: 5,
            name: "number_vmail_messages"
        },
        {
            id: 6,
            name: "total_day_minutes"
        },
        {
            id: 7,
            name: "total_day_calls"
        },
        {
            id: 8,
            name: "total_day_charge"
        },
        {
            id: 9,
            name: "total_eve_minutes"
        },
        {
            id: 10,
            name: "total_eve_calls"
        },
        {
            id: 11,
            name: "total_eve_charge"
        },
        {
            id: 12,
            name: "total_night_minutes"
        },
        {
            id: 13,
            name: "total_night_calls"
        },
        {
            id: 14,
            name: "total_night_charge"
        },
        {
            id: 15,
            name: "total_intl_minutes"
        },
        {
            id: 16,
            name: "total_intl_calls"
        },
        {
            id: 17,
            name: "total_intl_charge"
        },
        {
            id: 18,
            name: "number_customer_service_calls"
        },
    ]
    var features = []
    feature_names.forEach(feature => {
        features.push(<FeatureList key={feature.name} />)
    })

    return (
        <main className='main'>
            <div className='container'>
                <Header1 className={"display-1"} children={"Customer Churn Analysis"} />
                <Header2 className={"mt-5"} children={"Paste Excel row here."}></Header2>
                <Form className='d-flex flex-row form-floating align-items-center'>
                    <Input id={"pasteData"}  className={"w-75 form-control d-flex"} type="text" placeholder="Paste Excel row here" />
                    <Label forLabel={"pasteData"} children={"Paste Excel row here"} />
                    <Button className={"d-flex ms-3 h-50 inline"}>Scatter Values</Button>
                </Form>
                <Header2 className={"mt-3"} children={"...Or input below each value individually"}></Header2>
                <Form id={"predictionForm"} className={"form-floating"} action={"/predict"} method={"post"}>
                    {/* <Input id={"state"} className={"form-control"} type={"text"} placeholder={"state"} required={"required"} />
                    <Label forLabel={"state"} children={"State"} /> */}
                    {features}
                </Form>
            </div>
        </main>
    )
}

export default App