import React, { useState, useEffect } from 'react'
import Button from './Components/Button'
import Header1 from './Components/Header1'
import Input from './Components/Input'

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

    return (
        <main className='main'>
            <div className='container'>
                <Header1 children={"Customer Churn Analysis"} className="display-1" />
                <div className='d-flex flex-row'>
                    <Input id="pasteData"  className={"w-75 form-control d-flex"} type="text" placeholder="Paste Excel row here" />
                    <Button className={"d-flex align ms-3"}>Scatter Values</Button>
                </div>
            </div>
        </main>
    )
}

export default App