import React from 'react'
import Button from '../Components/Button'
import Header1 from '../Components/Header1'
import { Link } from 'react-router-dom'

function Home() {

    return (
        <>
            <div className='container' style={{ height: "480px" }}>
                <div className='d-flex flex-column text-center h-100'>
                    <div>
                        <Header1 className={'display-1'}>Welcome!</Header1>
                        <blockquote className="blockquote">
                            <p>"Unlocking Customer Retention: Empowering Businesses to Thrive Through Data-driven Churn Analysis."</p>
                        </blockquote>
                        <figcaption className="blockquote-footer">
                            Founders of <cite title="Source Title">TelCo Generic</cite>
                        </figcaption>
                        <Link to="/single_customer"><Button theme={'primary'}>Get Started!</Button></Link>
                    </div>
                </div>
            </div>
        </>
    )
}

export default Home